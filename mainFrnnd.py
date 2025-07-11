from datetime import datetime, date
from statistics import mean, stdev
from typing import List

import calendar
import logging
import numpy as np
import models
import models
import requests
from models import MetricasTiempo

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Path
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import SessionLocal, engine, get_db
from ml_predictor import modelo_ml
from models import (
    VentaInput,
    PrediccionResponse,
    EstadisticasMLResponse,
    PrediccionAlmacenadaInput,
    PrediccionAlmacenadaResponse,
    AnalisisHistoricoResponse,
    GenerarPrediccionesMasivasInput,
    MetricasTiempo,
    Comercializacion,
    Estado,
    Factura,
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear tablas (si no existen)
models.Base.metadata.create_all(bind=engine)

# Instancia principal de FastAPI
app = FastAPI(
    title="Backend Para la tabla Metricas Tiempo", 
    description="Este main es un test",
    version="3.0"
)

# Configurar CORS (si es necesario)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ENDPOINTS DELTAS DE TIEMPO (TESTS) =====
@app.get("/clientes")
def listar_clientes(db: Session = Depends(get_db)):
    """Lista todos los clientes únicos"""
    clientes = db.query(models.Comercializacion.Cliente).distinct().all()
    lista_clientes = [cliente[0] for cliente in clientes if cliente[0]]
    
    return {
        "clientes": sorted(lista_clientes),
        "total_clientes": len(lista_clientes)
    }

@app.get("/clientes/top")
def top_clientes_comercializaciones(db: Session = Depends(get_db)):
    """
    Devuelve el top 50 clientes con mayor número de comercializaciones,
    incluyendo idCliente y nombre.
    """
    resultados = (
        db.query(
            models.Comercializacion.ClienteId,
            models.Comercializacion.Cliente,
            func.count(models.Comercializacion.id).label("cantidad")
        )
        .filter(models.Comercializacion.Cliente.isnot(None))
        .group_by(models.Comercializacion.ClienteId, models.Comercializacion.Cliente)
        .order_by(func.count(models.Comercializacion.id).desc())
        .limit(50)
        .all()
    )

    top_clientes = [
        {
            "cliente_id": r[0],
            "cliente": r[1],
            "comercializaciones": r[2]
        }
        for r in resultados if r[0] and r[1]
    ]

    return {
        "top_50_clientes": top_clientes,
        "total": len(top_clientes)
    }

@app.get("/metricas/por_cliente")
def obtener_metricas_generales(db: Session = Depends(get_db)):
    """
    Devuelve un JSON con deltas generales por cliente (586 clientes únicos del dataset)
    """
    clientes = (
        db.query(Comercializacion.ClienteId, Comercializacion.Cliente)
        .filter(Comercializacion.Cliente.isnot(None))
        .distinct()
        .all()
    )

    resultados = []

    for id_cliente, nombre_cliente in clientes:
        comercializaciones = (
            db.query(Comercializacion)
            .filter(Comercializacion.ClienteId == id_cliente)
            .all()
        )

        delta_x_vals, delta_y_vals, delta_z_vals, delta_g_vals = [], [], [], []

        for com in comercializaciones:
            estados = db.query(Estado).filter(Estado.idComercializacion == com.id).all()
            facturas = db.query(Factura).filter(Factura.idComercializacion == com.id).all()

            # === ΔX ===
            estado_terminado = next((e for e in estados if e.EstadoComercializacion in [1, 3]), None)
            if com.FechaInicio and estado_terminado and estado_terminado.Fecha:
                dx = (estado_terminado.Fecha - com.FechaInicio).days
                if dx >= 0:
                    delta_x_vals.append(dx)

            # === ΔY ===
            if estado_terminado and facturas:
                fecha_estado = estado_terminado.Fecha
                primera_factura = min(facturas, key=lambda f: f.FechaFacturacion or date.max)
                if fecha_estado and primera_factura.FechaFacturacion:
                    dy = (primera_factura.FechaFacturacion - fecha_estado).days
                    if dy >= 0:
                        delta_y_vals.append(dy)

            # === ΔZ ===
            if facturas:
                pagadas = [f for f in facturas if f.EstadoFactura in [3, 4] and f.Pagado and f.Pagado > 0]
                if pagadas:
                    primera_factura = min(facturas, key=lambda f: f.FechaFacturacion or date.max)
                    ultima_pagada = max(pagadas, key=lambda f: f.FechaFacturacion or date.min)
                    if primera_factura.FechaFacturacion and ultima_pagada.FechaFacturacion:
                        dz = (ultima_pagada.FechaFacturacion - primera_factura.FechaFacturacion).days
                        if dz >= 0:
                            delta_z_vals.append(dz)

            # === ΔG ===
            if com.FechaInicio and facturas:
                pagadas = [f for f in facturas if f.EstadoFactura in [3, 4] and f.Pagado and f.Pagado > 0]
                if pagadas:
                    ultima_pagada = max(pagadas, key=lambda f: f.FechaFacturacion or date.min)
                    if ultima_pagada.FechaFacturacion:
                        dg = (ultima_pagada.FechaFacturacion - com.FechaInicio).days
                        if dg >= 0:
                            delta_g_vals.append(dg)

        resultados.append({
            "cliente_id": id_cliente,
            "cliente": nombre_cliente,
            "delta_x": round(sum(delta_x_vals) / len(delta_x_vals), 2) if delta_x_vals else None,
            "delta_y": round(sum(delta_y_vals) / len(delta_y_vals), 2) if delta_y_vals else None,
            "delta_z": round(sum(delta_z_vals) / len(delta_z_vals), 2) if delta_z_vals else None,
            "delta_g": round(sum(delta_g_vals) / len(delta_g_vals), 2) if delta_g_vals else None
        })

    return {"metricas_generales": resultados}

# ===== ENDPOINTS DELTAS DE TIEMPO =====

#ΔX: 0 - 1 entre comerc.
#ΔY: 1 comerc. - factura n1
#ΔZ: factura n1 - factura n (state:3/4 = con pago Y Pagado > 0)
#ΔG: 0 comerc. - factura n (state:3/4 = con pago Y Pagado > 0)
def calcular_metricas(valores: List[int]):
    if not valores:
        return {"promedio_dias": None, "desviacion_estandar": None, "cantidad": 0}
    return {
        "promedio_dias": round(mean(valores), 2),
        "desviacion_estandar": round(stdev(valores), 2) if len(valores) > 1 else 0,
        "cantidad": len(valores)
    }

@app.get("/api/deltaX")
@app.get("/api/deltaX/cliente/{cliente_id}")
def delta_x(cliente_id: int = None, db: Session = Depends(get_db)):
    """ΔX: Tiempo desde inicio hasta estado 'terminada' o 'terminada SENCE'"""
    query = db.query(models.Estado, models.Comercializacion).join(models.Comercializacion, models.Estado.idComercializacion == models.Comercializacion.id)
    if cliente_id:
        query = query.filter(models.Comercializacion.ClienteId == cliente_id)
    estados = query.filter(models.Estado.EstadoComercializacion.in_([1, 3])).all()

    valores = []
    for estado, com in estados:
        if com.FechaInicio and estado.Fecha:
            dias = (estado.Fecha - com.FechaInicio).days
            if dias >= 0:
                valores.append(dias)

    return calcular_metricas(valores)

@app.get("/api/deltaY")
@app.get("/api/deltaY/cliente/{cliente_id}")
def delta_y(cliente_id: int = None, db: Session = Depends(get_db)):
    """ΔY: Desde cuando el estado de comercio es 1 hasta la fecha de facturación de la primera factura"""
    comercializaciones = db.query(models.Comercializacion)
    if cliente_id:
        comercializaciones = comercializaciones.filter(models.Comercializacion.ClienteId == cliente_id)

    valores = []
    for com in comercializaciones:
        # Buscar el estado 1 (comercialización terminada) para esta comercialización
        estados_1 = db.query(models.Estado).filter(
            models.Estado.idComercializacion == com.id,
            models.Estado.EstadoComercializacion == 1
        ).all()
        
        if not estados_1:
            continue
        
        # Buscar facturas de esta comercialización
        facturas = db.query(models.Factura).filter(
            models.Factura.idComercializacion == com.id,
            models.Factura.FechaFacturacion.isnot(None)
        ).all()
        
        if not facturas:
            continue
        
        # Tomar la fecha del estado 1 (puede haber varios, tomar el último)
        fecha_estado_1 = max(estados_1, key=lambda e: e.Fecha).Fecha
        
        # Buscar la primera factura por fecha de facturación
        primera_factura = min(facturas, key=lambda f: f.FechaFacturacion)

        if fecha_estado_1 and primera_factura.FechaFacturacion:
            dias = (primera_factura.FechaFacturacion - fecha_estado_1).days
            if dias >= 0:
                valores.append(dias)

    return calcular_metricas(valores)

@app.get("/api/deltaZ")
@app.get("/api/deltaZ/cliente/{cliente_id}")
def delta_z(cliente_id: int = None, db: Session = Depends(get_db)):
    """ΔZ: Desde la fecha de facturación de la primera factura hasta la fecha del último pago real (estado 3/4 y Pagado > 0)"""
    comercializaciones = db.query(models.Comercializacion)
    if cliente_id:
        comercializaciones = comercializaciones.filter(models.Comercializacion.ClienteId == cliente_id)

    valores = []
    for com in comercializaciones:
        # Buscar facturas de esta comercialización
        facturas = db.query(models.Factura).filter(
            models.Factura.idComercializacion == com.id,
            models.Factura.FechaFacturacion.isnot(None)
        ).all()
        
        if not facturas:
            continue
        
        # Buscar la primera factura por fecha de facturación
        primera_factura = min(facturas, key=lambda f: f.FechaFacturacion)
        
        # Buscar facturas con estado de pago (3 o 4 = pagadas) Y que tengan pago real (!=0)
        facturas_pagadas = [f for f in facturas 
                           if f.EstadoFactura in [3, 4] 
                           and f.Pagado is not None 
                           and f.Pagado > 0]
        
        if facturas_pagadas:
            # Tomar la fecha del último pago (factura pagada más reciente)
            ultima_factura_pagada = max(facturas_pagadas, key=lambda f: f.FechaFacturacion)
            
            dias = (ultima_factura_pagada.FechaFacturacion - primera_factura.FechaFacturacion).days
            if dias >= 0:
                valores.append(dias)

    return calcular_metricas(valores)

@app.get("/api/deltaG")
@app.get("/api/deltaG/cliente/{cliente_id}")
def delta_g(cliente_id: int = None, db: Session = Depends(get_db)):
    """ΔG: Desde el estado de comercio 0 (inicio) hasta la fecha del último pago real (estado 3/4 y Pagado > 0)"""
    comercializaciones = db.query(models.Comercializacion)
    if cliente_id:
        comercializaciones = comercializaciones.filter(models.Comercializacion.ClienteId == cliente_id)

    valores = []
    for com in comercializaciones:
        # Verificar que tenga fecha de inicio
        if not com.FechaInicio:
            continue
        
        # Buscar facturas de esta comercialización con estado de pago (3 o 4 = pagadas) Y que tengan pago real (!=0)
        facturas_pagadas = db.query(models.Factura).filter(
            models.Factura.idComercializacion == com.id,
            models.Factura.EstadoFactura.in_([3, 4]),
            models.Factura.FechaFacturacion.isnot(None),
            models.Factura.Pagado.isnot(None),
            models.Factura.Pagado > 0
        ).all()
        
        if facturas_pagadas:
            # Tomar la fecha del último pago (factura pagada más reciente)
            ultima_factura_pagada = max(facturas_pagadas, key=lambda f: f.FechaFacturacion)
            
            dias = (ultima_factura_pagada.FechaFacturacion - com.FechaInicio).days
            if dias >= 0:
                valores.append(dias)

    return calcular_metricas(valores)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)