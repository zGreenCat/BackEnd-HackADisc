"""
🚀 BACKEND PRINCIPAL - HACKADISC
API FastAPI con endpoints de datos y predicciones ML
"""

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from sqlalchemy.orm import Session
from datetime import datetime
import logging
import numpy as np
import calendar
# Imports locales
from database import SessionLocal, engine
import models
from models import VentaInput, PrediccionResponse, EstadisticasMLResponse, PrediccionAlmacenadaInput, PrediccionAlmacenadaResponse, AnalisisHistoricoResponse, GenerarPrediccionesMasivasInput
from ml_predictor import modelo_ml
from statistics import mean, stdev
from fastapi import Path
from sqlalchemy import func
from database import get_db
import calendar
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear tablas (si no existen)
models.Base.metadata.create_all(bind=engine)

# Instancia principal de FastAPI
app = FastAPI(
    title="Backend INSECAP + Predicciones ML", 
    description="API completa para gestión de datos empresariales y predicciones de días de pago",
    version="3.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache en memoria para predicciones (en producción usar base de datos)
predicciones_cache = []

# ===== FUNCIONES AUXILIARES =====

def guardar_prediccion_cache(input_data: dict, response_data: dict):
    """Guarda predicción en cache como background task"""
    predicciones_cache.append({
        "timestamp": datetime.now(),
        "input": input_data,
        "respuesta": response_data
    })
    
    # Mantener solo últimas 1000 predicciones
    if len(predicciones_cache) > 1000:
        predicciones_cache.pop(0)

def validar_cliente_pagado_completo(comercializacion_id: int, db: Session) -> bool:
    """
    Valida si un cliente pagó completamente una comercialización específica.
    
    Criterios EXACTOS basados en la implementación de tu compañero:
    1. Estado de comercialización más reciente = 1 (terminada)
    2. Factura con estado de fecha más alta y tenga estado 3 y Pagado > 0  
    3. Suma total de todos los pagos >= valor de la comercialización
    
    Args:
        comercializacion_id: ID de la comercialización a validar
        db: Sesión de base de datos
        
    Returns:
        bool: True si el cliente pagó completamente, False en caso contrario
    """
    try:
        # Obtener la comercialización
        comercializacion = db.query(models.Comercializacion).filter(
            models.Comercializacion.id == comercializacion_id
        ).first()
        
        if not comercializacion or not comercializacion.ValorVenta:
            return False
        
        valor_venta = comercializacion.ValorVenta
        
        # 1. Verificar que el estado más reciente de la comercialización sea 1
        estados = db.query(models.Estado).filter(
            models.Estado.idComercializacion == comercializacion_id,
            models.Estado.Fecha.isnot(None)
        ).all()
        
        if not estados:
            return False
            
        # Encontrar el estado más reciente
        estado_mas_reciente = max(estados, key=lambda e: e.Fecha)
        if estado_mas_reciente.EstadoComercializacion != 1:
            return False
        
        # 2. Obtener todas las facturas de esta comercialización
        facturas = db.query(models.Factura).filter(
            models.Factura.idComercializacion == comercializacion_id
        ).all()
        
        if not facturas:
            return False
        
        # 3. Buscar la factura con estado de fecha más alta y verificar que tenga estado 3 y Pagado > 0
        facturas_validas = []
        fecha_estado_maxima = None
        factura_estado_3_pagada = False
        
        for factura in facturas:
            # Verificar si esta factura tiene estado 3 con pago > 0
            if (factura.EstadoFactura == 3 and 
                factura.Pagado and factura.Pagado > 0 and 
                factura.FechaEstado):
                
                if fecha_estado_maxima is None or factura.FechaEstado > fecha_estado_maxima:
                    fecha_estado_maxima = factura.FechaEstado
                    factura_estado_3_pagada = True
            
            # Reunir todas las facturas con pagos > 0 para sumatoria
            if factura.Pagado and factura.Pagado > 0:
                facturas_validas.append(factura.Pagado)
        
        # 4. Si hay factura válida con estado 3 pagado, y suma de pagos >= valor venta
        if factura_estado_3_pagada and sum(facturas_validas) >= valor_venta:
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error validando cliente pagado completo para comercialización {comercializacion_id}: {e}")
        return False

def obtener_comercializaciones_filtradas(db: Session, filtros_adicionales: dict = None, limite: int = None):
    """
    Obtiene comercializaciones aplicando todos los filtros necesarios:
    1. Excluir códigos ADI, OTR, SPD
    2. Solo clientes que pagaron completamente
    
    Args:
        db: Sesión de base de datos
        filtros_adicionales: Filtros adicionales a aplicar
        limite: Límite de resultados
        
    Returns:
        Lista de comercializaciones filtradas
    """
    query = db.query(models.Comercializacion).filter(
        ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
        ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
        ~models.Comercializacion.CodigoCotizacion.like('SPD%')
    )
    
    # Aplicar filtros adicionales si existen
    if filtros_adicionales:
        for campo, valor in filtros_adicionales.items():
            if hasattr(models.Comercializacion, campo):
                query = query.filter(getattr(models.Comercializacion, campo) == valor)
    
    if limite:
        query = query.limit(limite)
    
    comercializaciones = query.all()
    
    # Filtrar por clientes que pagaron completamente
    comercializaciones_pagadas = []
    for com in comercializaciones:
        if validar_cliente_pagado_completo(com.id, db):
            comercializaciones_pagadas.append(com)
    
    return comercializaciones_pagadas

# ===== ENDPOINTS PRINCIPALES =====

@app.get("/")
def root():
    """Endpoint raíz con información de la API completa"""
    return {
        "message": "Backend INSECAP + Predicciones ML activo",
        "version": "3.0",
        "proyecto": "HACKADISC 2025",
        "funcionalidades": {
            "datos": "CRUD completo de comercializaciones, facturas y estados",
            "ml": "Predicciones de días de pago" if modelo_ml.esta_disponible() else "ML no disponible",
            "documentacion": "/docs"
        },
        "modelo_ml": {
            "activo": modelo_ml.esta_disponible(),
            "mae": modelo_ml.metadata_hibrido.get('mae', 'N/A') if modelo_ml.metadata_hibrido else 'N/A',
            "r2": modelo_ml.metadata_hibrido.get('r2', 'N/A') if modelo_ml.metadata_hibrido else 'N/A'
        }
    }

@app.get("/health")
def health_check():
    """Endpoint de health check del sistema"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "database": "connected",
        "modelo_ml": "loaded" if modelo_ml.esta_disponible() else "not_available"
    }

# ===== ENDPOINTS DE DATOS =====

@app.get("/resumen")
def resumen(db: Session = Depends(get_db)):
    """Devuelve un resumen general con el total de registros cargados (filtrar ADI, OTR, SPD y solo clientes que pagaron)"""
    # Obtener comercializaciones base (sin filtro de pago completo)
    comercializaciones_base = db.query(models.Comercializacion).filter(
        ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
        ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
        ~models.Comercializacion.CodigoCotizacion.like('SPD%')
    ).all()
    
    # Filtrar solo las completamente pagadas
    total_ventas = 0
    for com in comercializaciones_base:
        if validar_cliente_pagado_completo(com.id, db):
            total_ventas += 1
    
    total_facturas = db.query(models.Factura).count()
    total_estados = db.query(models.Estado).count()

    return {
        "comercializaciones": total_ventas,
        "facturas": total_facturas,
        "estados": total_estados,
        "total_general": total_ventas + total_facturas + total_estados
    }

@app.get("/comercializaciones")
def obtener_comercializaciones(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    """Devuelve comercializaciones con paginación (filtrar ADI, OTR, SPD y solo clientes que pagaron)"""
    # Obtener comercializaciones base
    comercializaciones_base = db.query(models.Comercializacion).filter(
        ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
        ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
        ~models.Comercializacion.CodigoCotizacion.like('SPD%')
    ).offset(offset).limit(limit * 3).all()  # Obtener más para compensar el filtro
    
    # Filtrar solo las completamente pagadas
    comercializaciones = []
    for com in comercializaciones_base:
        if validar_cliente_pagado_completo(com.id, db):
            comercializaciones.append(com)
            if len(comercializaciones) >= limit:
                break
    
    # Calcular total real
    todas_comercializaciones = db.query(models.Comercializacion).filter(
        ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
        ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
        ~models.Comercializacion.CodigoCotizacion.like('SPD%')
    ).all()
    
    total = 0
    for com in todas_comercializaciones:
        if validar_cliente_pagado_completo(com.id, db):
            total += 1
    
    return {
        "comercializaciones": comercializaciones,
        "total": total,
        "limit": limit,
        "offset": offset
    }

@app.get("/cliente/{nombre}")
def obtener_cliente(nombre: str, db: Session = Depends(get_db)):
    """Devuelve todas las comercializaciones asociadas a un cliente por nombre exacto (filtrar ADI, OTR, SPD)"""
    resultado = db.query(models.Comercializacion).filter(
        models.Comercializacion.Cliente == nombre,
        ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
        ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
        ~models.Comercializacion.CodigoCotizacion.like('SPD%')
    ).all()
    if not resultado:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return {
        "cliente": nombre,
        "comercializaciones": resultado,
        "total_comercializaciones": len(resultado)
    }

@app.get("/facturas")
def obtener_facturas(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    """Devuelve facturas con paginación"""
    facturas = db.query(models.Factura).offset(offset).limit(limit).all()
    total = db.query(models.Factura).count()
    
    return {
        "facturas": facturas,
        "total": total,
        "limit": limit,
        "offset": offset
    }

@app.get("/estados")
def obtener_estados(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    """Devuelve estados con paginación"""
    estados = db.query(models.Estado).offset(offset).limit(limit).all()
    total = db.query(models.Estado).count()
    
    return {
        "estados": estados,
        "total": total,
        "limit": limit,
        "offset": offset
    }

@app.get("/sence")
def contar_sence(db: Session = Depends(get_db)):
    """Devuelve la cantidad de comercializaciones SENCE vs no-SENCE (filtrar ADI, OTR, SPD)"""
    total_sence = db.query(models.Comercializacion).filter(
        models.Comercializacion.EsSENCE == 1,
        ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
        ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
        ~models.Comercializacion.CodigoCotizacion.like('SPD%')
    ).count()
    total_no_sence = db.query(models.Comercializacion).filter(
        models.Comercializacion.EsSENCE == 0,
        ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
        ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
        ~models.Comercializacion.CodigoCotizacion.like('SPD%')
    ).count()

    return {
        "sence": total_sence,
        "no_sence": total_no_sence,
        "total": total_sence + total_no_sence,
        "porcentaje_sence": round((total_sence / (total_sence + total_no_sence)) * 100, 2)
    }

# ===== ENDPOINTS DE MACHINE LEARNING =====

@app.post("/predecir", response_model=PrediccionResponse)
def predecir_dias_pago(venta: VentaInput, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Predice los días hasta pago completo de una venta usando el modelo híbrido"""
    if not modelo_ml.esta_disponible():
        raise HTTPException(status_code=503, detail="Modelo de ML no disponible")
    
    try:
        # Preparar datos para el modelo
        datos_venta = venta.dict()
        
        # Realizar predicción
        resultado = modelo_ml.predecir_dias_pago(datos_venta)
        
        # Crear respuesta
        response = PrediccionResponse(**resultado)
        
        # Guardar en cache (background task)
        background_tasks.add_task(
            guardar_prediccion_cache,
            venta.dict(),
            response.dict()
        )
        
        logger.info(f"Predicción realizada: {resultado['dias_predichos']} días para cliente {venta.cliente}")
        return response
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno en predicción: {str(e)}")

@app.post("/predecir_lote")
def predecir_lote(ventas: List[VentaInput], background_tasks: BackgroundTasks):
    """Predice múltiples ventas en lote (máximo 50 por request)"""
    if not modelo_ml.esta_disponible():
        raise HTTPException(status_code=503, detail="Modelo de ML no disponible")
    
    if len(ventas) > 50:
        raise HTTPException(status_code=400, detail="Máximo 50 ventas por lote")
    
    resultados = []
    errores = 0
    
    for i, venta in enumerate(ventas):
        try:
            datos_venta = venta.dict()
            resultado = modelo_ml.predecir_lote_simplificado(datos_venta)
            
            resultado_final = {
                "indice": i,
                "cliente": venta.cliente,
                **resultado
            }
            resultados.append(resultado_final)
            
        except Exception as e:
            errores += 1
            resultados.append({
                "indice": i,
                "cliente": venta.cliente,
                "error": str(e)
            })
    
    return {
        "resultados": resultados, 
        "total_procesadas": len(ventas),
        "exitosas": len(ventas) - errores,
        "errores": errores,
        "timestamp": datetime.now()
    }

@app.get("/estadisticas_ml", response_model=EstadisticasMLResponse)
def obtener_estadisticas_ml():
    """Obtiene estadísticas de predicciones realizadas y información del modelo"""
    if not predicciones_cache:
        return EstadisticasMLResponse(
            total_predicciones=0,
            promedio_dias=0,
            distribuciones_riesgo={},
            modelo_info={"status": "sin_predicciones"},
            timestamp=datetime.now()
        )
    
    dias_predichos = [p['respuesta']['dias_predichos'] for p in predicciones_cache]
    riesgos = [p['respuesta']['codigo_riesgo'] for p in predicciones_cache]
    
    distribucion_riesgo = {}
    for riesgo in riesgos:
        distribucion_riesgo[riesgo] = distribucion_riesgo.get(riesgo, 0) + 1
    
    modelo_info = {
        "tipo": "Random Forest Híbrido",
        "mae": modelo_ml.metadata_hibrido.get('mae', 'N/A') if modelo_ml.metadata_hibrido else 'N/A',
        "r2": modelo_ml.metadata_hibrido.get('r2', 'N/A') if modelo_ml.metadata_hibrido else 'N/A',
        "features": len(modelo_ml.metadata_hibrido['features']) if modelo_ml.metadata_hibrido else 0,
        "fecha_entrenamiento": modelo_ml.metadata_hibrido.get('fecha_entrenamiento', 'N/A') if modelo_ml.metadata_hibrido else 'N/A'
    }
    
    return EstadisticasMLResponse(
        total_predicciones=len(predicciones_cache),
        promedio_dias=np.mean(dias_predichos),
        distribuciones_riesgo=distribucion_riesgo,
        modelo_info=modelo_info,
        timestamp=datetime.now()
    )

@app.get("/modelo/info")
def info_modelo():
    """Información detallada del modelo de ML"""
    if not modelo_ml.esta_disponible():
        raise HTTPException(status_code=503, detail="Modelo de ML no disponible")
    
    return modelo_ml.obtener_info_modelo()

@app.get("/modelo/test")
def test_modelo():
    """Endpoint para probar rápidamente el modelo con datos de ejemplo"""
    if not modelo_ml.esta_disponible():
        raise HTTPException(status_code=503, detail="Modelo de ML no disponible")
    
    # Datos de ejemplo
    ejemplo_datos = {
        "cliente": "CLIENTE_EJEMPLO",
        "correo_creador": "test@insecap.cl",
        "valor_venta": 500000,
        "es_sence": False,
        "mes_facturacion": 7,
        "cantidad_facturas": 2
    }
    
    try:
        resultado = modelo_ml.predecir_dias_pago(ejemplo_datos)
        
        return {
            "test_exitoso": True,
            "ejemplo_entrada": ejemplo_datos,
            "resultado": {
                "dias_predichos": resultado["dias_predichos"],
                "nivel_riesgo": resultado["nivel_riesgo"],
                "descripcion": resultado["descripcion_riesgo"],
                "accion": resultado["accion_recomendada"]
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        return {
            "test_exitoso": False,
            "error": str(e),
            "timestamp": datetime.now()
        }

# ===== ENDPOINTS ADICIONALES =====

@app.get("/clientes")
def listar_clientes(db: Session = Depends(get_db)):
    """Lista todos los clientes únicos que han pagado completamente (filtrar ADI, OTR, SPD)"""
    # Obtener todas las comercializaciones filtradas
    comercializaciones_base = db.query(models.Comercializacion).filter(
        ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
        ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
        ~models.Comercializacion.CodigoCotizacion.like('SPD%'),
        models.Comercializacion.Cliente.isnot(None),
        models.Comercializacion.ClienteId.isnot(None)
    ).distinct().all()
    
    # Filtrar solo clientes que han pagado completamente al menos una vez
    clientes_pagados = {}  # Usar dict para mantener ClienteId único
    clientes_por_nombre = {}  # Para contar por nombres únicos también
    
    for com in comercializaciones_base:
        if validar_cliente_pagado_completo(com.id, db):
            # Usar ClienteId como clave única (para el detalle)
            clientes_pagados[com.ClienteId] = {
                "cliente_id": com.ClienteId,
                "cliente_nombre": com.Cliente
            }
            # También agrupar por nombre para estadísticas
            if com.Cliente not in clientes_por_nombre:
                clientes_por_nombre[com.Cliente] = []
            clientes_por_nombre[com.Cliente].append(com.ClienteId)
    
    # Convertir a lista ordenada por nombre
    lista_clientes = sorted(list(clientes_pagados.values()), key=lambda x: x["cliente_nombre"])
    
    # Información sobre duplicados
    duplicados_info = {}
    for nombre, ids in clientes_por_nombre.items():
        if len(ids) > 1:
            duplicados_info[nombre] = {
                "cantidad_ids": len(ids),
                "client_ids": ids
            }
    
    return {
        "clientes": lista_clientes,
        "total_clientes": len(lista_clientes),
        "estadisticas": {
            "clientes_por_id_unico": len(clientes_pagados),
            "clientes_por_nombre_unico": len(clientes_por_nombre),
            "duplicados_por_nombre": len(duplicados_info),
            "detalle_duplicados": duplicados_info
        }
    }

@app.get("/clientes/top")
def top_clientes_comercializaciones(db: Session = Depends(get_db)):
    """
    Devuelve el top 50 clientes con mayor número de comercializaciones,
    incluyendo idCliente y nombre (filtrar ADI, OTR, SPD).
    """
    resultados = (
        db.query(
            models.Comercializacion.ClienteId,
            models.Comercializacion.Cliente,
            func.count(models.Comercializacion.id).label("cantidad")
        )
        .filter(
            models.Comercializacion.Cliente.isnot(None),
            ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
            ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
            ~models.Comercializacion.CodigoCotizacion.like('SPD%')
        )
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

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import MetricasTiempo, Comercializacion, Estado, Factura
from datetime import date
from collections import defaultdict

router = APIRouter()

@router.get("/metricas/historicas")
def obtener_metricas_historicas(db: Session = Depends(get_db)):
    """
    Devuelve un JSON con deltas mensuales por cliente (para los 50 clientes en la tabla MetricasTiempo)
    """
    # Obtener clientes únicos de la tabla MetricasTiempo
    clientes = db.query(MetricasTiempo.idCliente, MetricasTiempo.Cliente).distinct().all()
    resultados = []

    for id_cliente, nombre_cliente in clientes:
        for año in [2023, 2024, 2025]:
            max_mes = 12 if año < 2025 else 3
            for mes in range(1, max_mes + 1):
                fecha_inicio = date(año, mes, 1)
                fecha_fin = date(año, mes, 28)  # Simplificamos

                # Obtener comercializaciones válidas del cliente en ese mes
                comercializaciones = (
                    db.query(Comercializacion)
                    .filter(Comercializacion.ClienteId == id_cliente)
                    .filter(Comercializacion.FechaInicio >= fecha_inicio)
                    .filter(Comercializacion.FechaInicio < date(año, mes + 1, 1) if mes < 12 else date(año + 1, 1, 1))
                    .all()
                )

                delta_x_vals, delta_y_vals, delta_z_vals, delta_g_vals = [], [], [], []

                for com in comercializaciones:
                    estados = db.query(Estado).filter(Estado.idComercializacion == com.id).all()
                    facturas = db.query(Factura).filter(Factura.idComercializacion == com.id).all()

                    # === Delta X ===
                    estado_terminado = next((e for e in estados if e.EstadoComercializacion in [1, 3]), None)
                    if com.FechaInicio and estado_terminado and estado_terminado.Fecha:
                        dx = (estado_terminado.Fecha - com.FechaInicio).days
                        if dx >= 0: delta_x_vals.append(dx)

                    # === Delta Y ===
                    if estado_terminado and facturas:
                        fecha_estado = estado_terminado.Fecha
                        primera_factura = min(facturas, key=lambda f: f.FechaFacturacion or date.max)
                        if fecha_estado and primera_factura.FechaFacturacion:
                            dy = (primera_factura.FechaFacturacion - fecha_estado).days
                            if dy >= 0: delta_y_vals.append(dy)

                    # === Delta Z ===
                    pagadas = [f for f in facturas if f.EstadoFactura in [3, 4] and f.Pagado and f.Pagado > 0]
                    if facturas and pagadas:
                        primera_factura = min(facturas, key=lambda f: f.FechaFacturacion or date.max)
                        ultima_pagada = max(pagadas, key=lambda f: f.FechaFacturacion or date.min)
                        if primera_factura.FechaFacturacion and ultima_pagada.FechaFacturacion:
                            dz = (ultima_pagada.FechaFacturacion - primera_factura.FechaFacturacion).days
                            if dz >= 0: delta_z_vals.append(dz)

                    # === Delta G ===
                    if com.FechaInicio and pagadas:
                        ultima_pagada = max(pagadas, key=lambda f: f.FechaFacturacion or date.min)
                        dg = (ultima_pagada.FechaFacturacion - com.FechaInicio).days
                        if dg >= 0: delta_g_vals.append(dg)

                # Resumen mensual
                resultados.append({
                    "cliente_id": id_cliente,
                    "cliente": nombre_cliente,
                    "año": año,
                    "mes": mes,
                    "delta_x": round(sum(delta_x_vals) / len(delta_x_vals), 2) if delta_x_vals else None,
                    "delta_y": round(sum(delta_y_vals) / len(delta_y_vals), 2) if delta_y_vals else None,
                    "delta_z": round(sum(delta_z_vals) / len(delta_z_vals), 2) if delta_z_vals else None,
                    "delta_g": round(sum(delta_g_vals) / len(delta_g_vals), 2) if delta_g_vals else None
                })

    return {"metricas_historicas": resultados}


# ===== ENDPOINT PARA PREDICCIONES DE INGRESOS MENSUALES =====

@app.get("/prediccion_ingresos/{ano}/{mes}")
def predecir_ingresos_mes(ano: int, mes: int, limite: int = 500, db: Session = Depends(get_db)):
    """
    Predice cuánto dinero se recibirá en un mes específico
    
    Args:
        ano: Año de predicción (ej: 2025)
        mes: Mes de predicción (1-12)  
        limite: Límite de comercializaciones a analizar (máximo 1000)
    
    Returns:
        Predicción detallada de ingresos para el mes especificado
    """
    if not modelo_ml.esta_disponible():
        raise HTTPException(status_code=503, detail="Modelo de ML no disponible")
    
    if not (1 <= mes <= 12):
        raise HTTPException(status_code=400, detail="Mes debe estar entre 1 y 12")
    
    if not (2024 <= ano <= 2030):
        raise HTTPException(status_code=400, detail="Año debe estar entre 2024 y 2030")
    
    limite = min(max(limite, 10), 1000)  # Entre 10 y 1000
    
    try:
        from predictor_universal import PredictorIngresosMensuales
        
        # Usar el predictor universal
        with PredictorIngresosMensuales() as predictor:
            resultado = predictor.predecir_ingresos_mes(ano, mes, limite)
            
        return resultado
        
    except Exception as e:
        logger.error(f"Error en predicción de ingresos: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/prediccion_ingresos_resumen/{ano}/{mes}")
def predecir_ingresos_resumen(ano: int, mes: int, db: Session = Depends(get_db)):
    """
    Versión resumida de predicción de ingresos (más rápida)
    
    Args:
        ano: Año de predicción (ej: 2025)
        mes: Mes de predicción (1-12)
    
    Returns:
        Resumen ejecutivo de la predicción
    """
    if not modelo_ml.esta_disponible():
        raise HTTPException(status_code=503, detail="Modelo de ML no disponible")
    
    if not (1 <= mes <= 12):
        raise HTTPException(status_code=400, detail="Mes debe estar entre 1 y 12")
    
    if not (2024 <= ano <= 2030):
        raise HTTPException(status_code=400, detail="Año debe estar entre 2024 y 2030")
    
    try:
        from predictor_universal import PredictorIngresosMensuales
        import calendar
        
        # Usar el predictor con límite reducido para velocidad
        with PredictorIngresosMensuales() as predictor:
            resultado = predictor.predecir_ingresos_mes(ano, mes, limite=100)
            
        # Retornar solo el resumen
        return {
            "periodo": f"{calendar.month_name[mes]} {ano}",
            "resumen_ejecutivo": {
                "valor_proyectado": resultado['resumen_principal']['valor_proyectado'],
                "cantidad_pagos": resultado['resumen_principal']['cantidad_pagos'],
                "porcentaje_del_pendiente": resultado['resumen_principal']['porcentaje_del_pendiente'],
                "confianza": resultado['resumen_principal']['confianza_promedio']
            },
            "contexto": {
                "valor_total_pendiente": resultado['contexto']['valor_total_pendiente'],
                "comercializaciones_pendientes": resultado['contexto']['total_comercializaciones_pendientes']
            },
            "sence": resultado['analisis_sence'],
            "fecha_generacion": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en predicción de ingresos resumen: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/cliente/{cliente_id}/probabilidad_pago_mes_actual")
def probabilidad_pago_mes_actual(cliente_id: int, db: Session = Depends(get_db)):
    """
    Predice la probabilidad de que un cliente pague dentro del mes actual,
    separando entre ventas SENCE y no SENCE.
    """
    if not modelo_ml.esta_disponible():
        raise HTTPException(status_code=503, detail="Modelo de ML no disponible")

    hoy = datetime.now()
    dias_restantes = (calendar.monthrange(hoy.year, hoy.month)[1] - hoy.day)

    # Obtener ventas activas del cliente (filtrar ADI, OTR, SPD)
    ventas = (
        db.query(models.Comercializacion)
        .filter(
            models.Comercializacion.ClienteId == cliente_id,
            models.Comercializacion.EstadoComercializacion.in_([0, 1, 3]),  # En proceso, terminado, SENCE
            ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
            ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
            ~models.Comercializacion.CodigoCotizacion.like('SPD%')
        )
        .all()
    )

    if not ventas:
        raise HTTPException(status_code=404, detail="No se encontraron ventas activas para el cliente")

    resultados_sence = []
    resultados_no_sence = []

    for venta in ventas:
        datos_venta = {
            "cliente": venta.Cliente,
            "correo_creador": venta.CorreoCreador,
            "valor_venta": venta.ValorFinalComercializacion,
            "es_sence": bool(venta.EsSENCE),
            "mes_facturacion": hoy.month,
            "cantidad_facturas": len(venta.Facturas or [])
        }

        try:
            prediccion = modelo_ml.predecir_dias_pago(datos_venta)
            dias = prediccion.get("dias_predichos", 99)
            pago_este_mes = dias <= dias_restantes

            resultado = {
                "idComercializacion": venta.id,
                "dias_predichos": dias,
                "pago_este_mes": pago_este_mes
            }

            if datos_venta["es_sence"]:
                resultados_sence.append(resultado)
            else:
                resultados_no_sence.append(resultado)

        except Exception as e:
            resultado = {
                "idComercializacion": venta.id,
                "error": str(e)
            }
            if datos_venta["es_sence"]:
                resultados_sence.append(resultado)
            else:
                resultados_no_sence.append(resultado)

    # Calcular probabilidades
    def calcular_probabilidad(resultados):
        válidos = [r for r in resultados if "pago_este_mes" in r]
        if not válidos:
            return 0.0
        positivos = sum(1 for r in válidos if r["pago_este_mes"])
        return round(positivos / len(válidos), 2)

    return {
        "cliente_id": cliente_id,
        "cliente": ventas[0].Cliente,
        "mes_actual": hoy.strftime("%B"),
        "probabilidades": {
            "SENCE": {
                "probabilidad_pago_mes": calcular_probabilidad(resultados_sence),
                "ventas_analizadas": len(resultados_sence),
                "detalles": resultados_sence
            },
            "No SENCE": {
                "probabilidad_pago_mes": calcular_probabilidad(resultados_no_sence),
                "ventas_analizadas": len(resultados_no_sence),
                "detalles": resultados_no_sence
            }
        },
        "fecha_consulta": hoy.isoformat()
    }

# ===== ENDPOINTS DE PROYECCIÓN DE VENTAS =====

@app.get("/proyeccion_anual/{ano}")
def calcular_proyeccion_ventas_anuales(
    ano: int, 
    incluir_predicciones: bool = True,
    limite_clientes: int = 1000,
    db: Session = Depends(get_db)
):
    """
    Calcula la proyección completa de ventas e ingresos para un año específico.
    Combina datos históricos reales con predicciones ML para meses futuros.
    
    Args:
        ano: Año a analizar (ej: 2025)
        incluir_predicciones: Si incluir predicciones ML para meses sin datos
        limite_clientes: Límite de clientes a procesar (máximo 2000)
    
    Returns:
        Proyección detallada mes a mes con datos reales y predicciones
    """
    if not (2020 <= ano <= 2030):
        raise HTTPException(status_code=400, detail="Año debe estar entre 2020 y 2030")
    
    limite_clientes = min(max(limite_clientes, 10), 2000)
    fecha_actual = datetime.now()
    es_ano_futuro = ano > fecha_actual.year
    mes_actual = fecha_actual.month if ano == fecha_actual.year else 12
    
    # Estructura de respuesta
    resultado_meses = {}
    resumen_anual = {
        "valor_total_real": 0,
        "valor_total_proyectado": 0,
        "meses_con_datos_reales": 0,
        "meses_proyectados": 0,
        "total_ventas_reales": 0,
        "total_ventas_proyectadas": 0
    }
    
    try:
        # 1. OBTENER DATOS HISTÓRICOS REALES
        for mes in range(1, 13):
            # Datos reales de comercializaciones del mes (filtrar códigos ADI, OTR, SPD y solo clientes que pagaron)
            filtros_mes = {
                'FechaInicio': f'EXTRACT(year FROM FechaInicio) = {ano} AND EXTRACT(month FROM FechaInicio) = {mes}'
            }
            comercializaciones_mes_base = db.query(models.Comercializacion).filter(
                func.extract('year', models.Comercializacion.FechaInicio) == ano,
                func.extract('month', models.Comercializacion.FechaInicio) == mes,
                models.Comercializacion.ValorVenta.isnot(None),
                ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
                ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
                ~models.Comercializacion.CodigoCotizacion.like('SPD%')
            ).limit(limite_clientes).all()
            
            # Filtrar solo las que están completamente pagadas
            comercializaciones_mes = []
            for com in comercializaciones_mes_base:
                if validar_cliente_pagado_completo(com.id, db):
                    comercializaciones_mes.append(com)
            
            # Datos reales de facturas pagadas del mes (filtrar códigos ADI, OTR, SPD y solo clientes que pagaron)
            facturas_pagadas_mes_base = db.query(models.Factura).join(
                models.Comercializacion, 
                models.Factura.idComercializacion == models.Comercializacion.id
            ).filter(
                func.extract('year', models.Factura.FechaFacturacion) == ano,
                func.extract('month', models.Factura.FechaFacturacion) == mes,
                models.Factura.EstadoFactura.in_([3, 4]),
                models.Factura.Pagado.isnot(None),
                models.Factura.Pagado > 0,
                ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
                ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
                ~models.Comercializacion.CodigoCotizacion.like('SPD%')
            ).all()
            
            # Filtrar solo facturas de comercializaciones completamente pagadas
            facturas_pagadas_mes = []
            for factura in facturas_pagadas_mes_base:
                if validar_cliente_pagado_completo(factura.idComercializacion, db):
                    facturas_pagadas_mes.append(factura)
            
            # Calcular valores reales
            valor_ventas_mes = sum(c.ValorVenta for c in comercializaciones_mes if c.ValorVenta)
            valor_cobrado_mes = sum(f.Pagado for f in facturas_pagadas_mes if f.Pagado)
            cantidad_ventas_mes = len(comercializaciones_mes)
            cantidad_pagos_mes = len(facturas_pagadas_mes)
            
            # Determinar si necesitamos predicciones
            tiene_datos_reales = cantidad_ventas_mes > 0 or cantidad_pagos_mes > 0
            necesita_prediccion = not tiene_datos_reales and incluir_predicciones
            
            # Inicializar datos del mes
            datos_mes = {
                "mes": mes,
                "nombre_mes": calendar.month_name[mes],
                "datos_reales": {
                    "valor_ventas": valor_ventas_mes,
                    "valor_cobrado": valor_cobrado_mes,
                    "cantidad_ventas": cantidad_ventas_mes,
                    "cantidad_pagos": cantidad_pagos_mes,
                    "tiene_datos": tiene_datos_reales
                },
                "predicciones": None,
                "resumen": {
                    "valor_total_estimado": valor_ventas_mes + valor_cobrado_mes,
                    "fuente": "datos_reales" if tiene_datos_reales else "sin_datos"
                }
            }
            
            # 2. AGREGAR PREDICCIONES ML SI ES NECESARIO
            if necesita_prediccion and modelo_ml.esta_disponible():
                try:
                    # Obtener clientes activos para predicciones (filtrar códigos ADI, OTR, SPD)
                    clientes_activos = db.query(
                        models.Comercializacion.ClienteId,
                        models.Comercializacion.Cliente,
                        models.Comercializacion.LiderComercial
                    ).filter(
                        models.Comercializacion.Cliente.isnot(None),
                        ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
                        ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
                        ~models.Comercializacion.CodigoCotizacion.like('SPD%')
                    ).distinct().limit(limite_clientes // 12).all()  # Distribuir clientes por mes
                    
                    predicciones_mes = []
                    valor_predicciones = 0
                    errores_prediccion = 0
                    
                    for cliente_id, cliente_nombre, lider in clientes_activos:
                        if not cliente_id or not cliente_nombre:
                            continue
                            
                        try:
                            # Calcular valor promedio histórico del cliente (filtrar códigos ADI, OTR, SPD)
                            ventas_historicas = db.query(models.Comercializacion.ValorVenta).filter(
                                models.Comercializacion.ClienteId == cliente_id,
                                models.Comercializacion.ValorVenta.isnot(None),
                                ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
                                ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
                                ~models.Comercializacion.CodigoCotizacion.like('SPD%')
                            ).all()
                            
                            if not ventas_historicas:
                                continue
                                
                            valor_promedio = sum(v[0] for v in ventas_historicas if v[0]) / len(ventas_historicas)
                            
                            # Determinar si es cliente SENCE
                            sence_check = db.query(models.Comercializacion.EsSENCE).filter(
                                models.Comercializacion.ClienteId == cliente_id
                            ).first()
                            es_sence = bool(sence_check[0]) if sence_check else False
                            
                            # Realizar predicción
                            datos_prediccion = {
                                "cliente": cliente_nombre,
                                "correo_creador": "sistema@insecap.cl",
                                "valor_venta": valor_promedio,
                                "es_sence": es_sence,
                                "mes_facturacion": mes,
                                "cantidad_facturas": 1
                            }
                            
                            resultado_pred = modelo_ml.predecir_dias_pago(datos_prediccion)
                            
                            # Solo considerar pagos que caen dentro del mes
                            dias_predichos = resultado_pred.get("dias_predichos", 99)
                            if dias_predichos <= 31:  # Se paga dentro del mes
                                valor_predicciones += valor_promedio
                                predicciones_mes.append({
                                    "cliente_id": cliente_id,
                                    "cliente_nombre": cliente_nombre,
                                    "valor_estimado": valor_promedio,
                                    "dias_predichos": dias_predichos,
                                    "nivel_riesgo": resultado_pred.get("nivel_riesgo", "UNKNOWN"),
                                    "es_sence": es_sence
                                })
                            
                        except Exception:
                            errores_prediccion += 1
                            logger.error(f"Error predicción cliente {cliente_id} mes {mes}: {e}")
                    
                    # Actualizar datos del mes con predicciones
                    if predicciones_mes:
                        datos_mes["predicciones"] = {
                            "valor_proyectado": valor_predicciones,
                            "cantidad_predicciones": len(predicciones_mes),
                            "errores": errores_prediccion,
                            "detalle_predicciones": predicciones_mes[:10],  # Solo primeras 10 para no sobrecargar
                            "modelo_version": "v3.0-hibrido"
                        }
                        datos_mes["resumen"]["valor_total_estimado"] += valor_predicciones
                        datos_mes["resumen"]["fuente"] = "predicciones_ml"
                        
                except Exception as e:
                    logger.error(f"Error general en predicciones para mes {mes}: {e}")
                    datos_mes["predicciones"] = {"error": str(e)}
            
            # 3. ACTUALIZAR RESÚMENES
            if tiene_datos_reales:
                resumen_anual["valor_total_real"] += valor_ventas_mes + valor_cobrado_mes
                resumen_anual["meses_con_datos_reales"] += 1
                resumen_anual["total_ventas_reales"] += cantidad_ventas_mes
            
            if datos_mes["predicciones"] and "valor_proyectado" in datos_mes["predicciones"]:
                resumen_anual["valor_total_proyectado"] += datos_mes["predicciones"]["valor_proyectado"]
                resumen_anual["meses_proyectados"] += 1
                resumen_anual["total_ventas_proyectadas"] += datos_mes["predicciones"]["cantidad_predicciones"]
            
            resultado_meses[f"mes_{mes:02d}"] = datos_mes
        
        # 4. CALCULAR MÉTRICAS FINALES
        valor_total_combinado = resumen_anual["valor_total_real"] + resumen_anual["valor_total_proyectado"]
        
        return {
            "ano": ano,
            "fecha_consulta": datetime.now(),
            "configuracion": {
                "incluir_predicciones": incluir_predicciones,
                "limite_clientes": limite_clientes,
                "modelo_ml_disponible": modelo_ml.esta_disponible()
            },
            "resumen_anual": {
                **resumen_anual,
                "valor_total_combinado": valor_total_combinado,
                "promedio_mensual": valor_total_combinado / 12 if valor_total_combinado > 0 else 0,
                "porcentaje_real": round((resumen_anual["valor_total_real"] / valor_total_combinado) * 100, 2) if valor_total_combinado > 0 else 0,
                "porcentaje_proyectado": round((resumen_anual["valor_total_proyectado"] / valor_total_combinado) * 100, 2) if valor_total_combinado > 0 else 0
            },
            "detalle_mensual": resultado_meses,
            "recomendaciones": {
                "precision": "Alta" if resumen_anual["meses_con_datos_reales"] >= 6 else "Media" if resumen_anual["meses_con_datos_reales"] >= 3 else "Baja",
                "confiabilidad": "Los datos históricos son definitivos. Las predicciones tienen un MAE de ~2.5 días.",
                "uso_sugerido": "Ideal para planificación financiera y proyecciones de flujo de caja"
            }
        }
        
    except Exception as e:
        logger.error(f"Error en proyección anual {ano}: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculando proyección anual: {str(e)}")

@app.get("/proyeccion_trimestral/{ano}/{trimestre}")
def calcular_proyeccion_trimestral(
    ano: int,
    trimestre: int,
    incluir_predicciones: bool = True,
    db: Session = Depends(get_db)
):
    """
    Calcula la proyección de ventas para un trimestre específico.
    Más detallado que la proyección anual.
    
    Args:
        ano: Año (ej: 2025)
        trimestre: Trimestre (1, 2, 3, 4)
        incluir_predicciones: Si incluir predicciones ML
    
    Returns:
        Proyección detallada del trimestre
    """
    if not (1 <= trimestre <= 4):
        raise HTTPException(status_code=400, detail="Trimestre debe estar entre 1 y 4")
    
    if not (2020 <= ano <= 2030):
        raise HTTPException(status_code=400, detail="Año debe estar entre 2020 y 2030")
    
    # Calcular meses del trimestre
    meses_trimestre = {
        1: [1, 2, 3],
        2: [4, 5, 6], 
        3: [7, 8, 9],
        4: [10, 11, 12]
    }
    
    meses = meses_trimestre[trimestre]
    
    try:
        resultado_detallado = {}
        totales_trimestre = {
            "valor_real": 0,
            "valor_proyectado": 0,
            "ventas_reales": 0,
            "ventas_proyectadas": 0
        }
        
        for mes in meses:
            # Datos reales del mes (filtrar códigos ADI, OTR, SPD)
            ventas_mes = db.query(models.Comercializacion).filter(
                func.extract('year', models.Comercializacion.FechaInicio) == ano,
                func.extract('month', models.Comercializacion.FechaInicio) == mes,
                models.Comercializacion.ValorVenta.isnot(None),
                ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
                ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
                ~models.Comercializacion.CodigoCotizacion.like('SPD%')
            ).all()
            
            pagos_mes = db.query(models.Factura).join(
                models.Comercializacion,
                models.Factura.idComercializacion == models.Comercializacion.id
            ).filter(
                func.extract('year', models.Factura.FechaFacturacion) == ano,
                func.extract('month', models.Factura.FechaFacturacion) == mes,
                models.Factura.EstadoFactura.in_([3, 4]),
                models.Factura.Pagado > 0,
                ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
                ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
                ~models.Comercializacion.CodigoCotizacion.like('SPD%')
            ).all()
            
            valor_ventas = sum(v.ValorVenta for v in ventas_mes if v.ValorVenta)
            valor_pagos = sum(p.Pagado for p in pagos_mes if p.Pagado)
            
            # Análisis por tipo de cliente
            ventas_sence = [v for v in ventas_mes if v.EsSENCE]
            ventas_comercial = [v for v in ventas_mes if not v.EsSENCE]
            
            datos_mes = {
                "mes": mes,
                "nombre_mes": calendar.month_name[mes],
                "datos_reales": {
                    "ventas_totales": valor_ventas,
                    "pagos_recibidos": valor_pagos,
                    "cantidad_ventas": len(ventas_mes),
                    "cantidad_pagos": len(pagos_mes)
                },
                "segmentacion": {
                    "sence": {
                        "cantidad": len(ventas_sence),
                        "valor": sum(v.ValorVenta for v in ventas_sence if v.ValorVenta)
                    },
                    "comercial": {
                        "cantidad": len(ventas_comercial),
                        "valor": sum(v.ValorVenta for v in ventas_comercial if v.ValorVenta)
                    }
                }
            }
            
            # Agregar predicciones si es necesario
            if incluir_predicciones and modelo_ml.esta_disponible() and len(ventas_mes) == 0:
                # Solo predecir si no hay datos reales
                try:
                    # Usar clientes históricos para predecir
                    clientes_sample = db.query(
                        models.Comercializacion.ClienteId,
                        models.Comercializacion.Cliente
                    ).distinct().limit(50).all()
                    
                    predicciones = []
                    for cliente_id, cliente_nombre in clientes_sample:
                        if not cliente_id:
                            continue
                            
                        # Valor promedio histórico
                        valor_hist = db.query(models.Comercializacion.ValorVenta).filter(
                            models.Comercializacion.ClienteId == cliente_id,
                            models.Comercializacion.ValorVenta.isnot(None)
                        ).all()
                        
                        if valor_hist:
                            valor_prom = sum(v[0] for v in valor_hist) / len(valor_hist)
                            
                            # Predicción
                            datos_pred = {
                                "cliente": cliente_nombre,
                                "correo_creador": "sistema@insecap.cl",
                                "valor_venta": valor_prom,
                                "es_sence": False,  # Simplificado
                                "mes_facturacion": mes,
                                "cantidad_facturas": 1
                            }
                            
                            pred_result = modelo_ml.predecir_dias_pago(datos_pred)
                            if pred_result.get("dias_predichos", 99) <= 31:
                                predicciones.append({
                                    "cliente": cliente_nombre,
                                    "valor": valor_prom,
                                    "dias": pred_result.get("dias_predichos")
                                })
                    
                   
                    
                    if predicciones:
                        datos_mes["predicciones"] = {
                            "valor_estimado": sum(p["valor"] for p in predicciones),
                            "cantidad": len(predicciones),
                            "detalle": predicciones[:5]  # Solo las primeras 5
                        }
                        totales_trimestre["valor_proyectado"] += datos_mes["predicciones"]["valor_estimado"]
                        totales_trimestre["ventas_proyectadas"] += len(predicciones)
                    
                except Exception as e:
                    datos_mes["predicciones"] = {"error": str(e)}
            
            # Actualizar totales
            totales_trimestre["valor_real"] += valor_ventas + valor_pagos
            totales_trimestre["ventas_reales"] += len(ventas_mes)
            
            resultado_detallado[f"mes_{mes}"] = datos_mes
        
        return {
            "ano": ano,
            "trimestre": trimestre,
            "nombre_trimestre": f"Q{trimestre} {ano}",
            "meses_incluidos": [calendar.month_name[m] for m in meses],
            "resumen_trimestre": totales_trimestre,
            "detalle_mensual": resultado_detallado,
            "fecha_consulta": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error en proyección trimestral {trimestre}/{ano}: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculando proyección trimestral: {str(e)}")

# ===== ENDPOINT SIMPLIFICADO DE PROYECCIÓN ANUAL =====

@app.get("/proyeccion_anual_simplificada/{ano}")
def calcular_proyeccion_anual_simplificada(
    ano: int,
    incluir_predicciones: bool = True,
    db: Session = Depends(get_db)
):
    """
    Proyección anual simplificada con desglose mensual claro.
    Devuelve el valor estimado para cada mes del año y el total anual.
    
    Args:
        ano: Año a analizar (ej: 2025)
        incluir_predicciones: Si incluir predicciones ML para meses futuros
    
    Returns:
        Proyección con valores mensuales y total anual
    """
    if not (2020 <= ano <= 2030):
        raise HTTPException(status_code=400, detail="Año debe estar entre 2020 y 2030")
    
    fecha_actual = datetime.now()
    
    # Estructura de respuesta
    valores_mensuales = {}
    total_anual = 0
    meses_con_datos = 0
    meses_proyectados = 0
    
    try:
        for mes in range(1, 13):
            mes_nombre = [
                "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
            ][mes - 1]
            
            # 1. DATOS REALES DEL MES - LÓGICA CORREGIDA
            # SOLO comercializaciones con FechaInicio en el mes que estén completamente pagadas
            comercializaciones_mes_base = db.query(models.Comercializacion).filter(
                func.extract('year', models.Comercializacion.FechaInicio) == ano,
                func.extract('month', models.Comercializacion.FechaInicio) == mes,
                models.Comercializacion.ValorVenta.isnot(None),
                ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
                ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
                ~models.Comercializacion.CodigoCotizacion.like('SPD%')
            ).all()
            
            # Procesar solo comercializaciones completamente pagadas
            valor_ventas_real = 0
            valor_cobrado_real = 0  # Suma de todos los pagos de las comercializaciones del mes
            cantidad_ventas_real = 0
            cantidad_pagos_real = 0
            
            for com in comercializaciones_mes_base:
                if validar_cliente_pagado_completo(com.id, db):
                    # Agregar valor de venta
                    valor_ventas_real += com.ValorVenta or 0
                    cantidad_ventas_real += 1
                    
                    # Agregar TODOS los pagos de esta comercialización (independiente de cuándo se pagaron)
                    facturas_com = db.query(models.Factura).filter(
                        models.Factura.idComercializacion == com.id,
                        models.Factura.Pagado.isnot(None),
                        models.Factura.Pagado > 0
                    ).all()
                    
                    for factura in facturas_com:
                        valor_cobrado_real += factura.Pagado or 0
                        cantidad_pagos_real += 1
            
            # Valor total real del mes (ventas + cobros)
            valor_real_mes = valor_ventas_real + valor_cobrado_real
            tiene_datos_reales = valor_real_mes > 0
            
            # 2. PREDICCIONES ML (si no hay datos reales y se solicitan)
            valor_prediccion_mes = 0
            cantidad_predicciones = 0
            fuente_datos = "sin_datos"
            
            if not tiene_datos_reales and incluir_predicciones and modelo_ml.esta_disponible():
                try:
                    # Obtener muestra de clientes activos para predicción
                    clientes_activos = db.query(
                        models.Comercializacion.ClienteId,
                        models.Comercializacion.Cliente
                    ).filter(
                        models.Comercializacion.Cliente.isnot(None),
                        models.Comercializacion.ClienteId.isnot(None),
                        ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
                        ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
                        ~models.Comercializacion.CodigoCotizacion.like('SPD%')
                    ).distinct().limit(100).all()  # Muestra representativa
                    
                    for cliente_id, cliente_nombre in clientes_activos:
                        try:
                            # Valor promedio histórico del cliente
                            ventas_historicas = db.query(models.Comercializacion.ValorVenta).filter(
                                models.Comercializacion.ClienteId == cliente_id,
                                models.Comercializacion.ValorVenta.isnot(None),
                                ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
                                ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
                                ~models.Comercializacion.CodigoCotizacion.like('SPD%')
                            ).all()
                            
                            if ventas_historicas:
                                valor_promedio = sum(v[0] for v in ventas_historicas if v[0]) / len(ventas_historicas)
                                
                                # Determinar si es cliente SENCE
                                sence_info = db.query(models.Comercializacion.EsSENCE).filter(
                                    models.Comercializacion.ClienteId == cliente_id,
                                    ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
                                    ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
                                    ~models.Comercializacion.CodigoCotizacion.like('SPD%')
                                ).first()
                                es_sence = bool(sence_info[0]) if sence_info else False
                                
                                # Datos para predicción con el mes específico
                                datos_prediccion = {
                                    "cliente": cliente_nombre,
                                    "correo_creador": "prediccion@insecap.cl",
                                    "valor_venta": valor_promedio,
                                    "es_sence": es_sence,
                                    "mes_facturacion": mes,  # Mes específico
                                    "cantidad_facturas": 1
                                }
                                
                                # Realizar predicción con el modelo mejorado
                                resultado_pred = modelo_ml.predecir_dias_pago(datos_prediccion)
                                dias_predichos = resultado_pred.get("dias_predichos", 99)
                                
                                # LÓGICA MEJORADA: Calcular probabilidad de pago en el mes
                                # Simular que las ventas se hacen al principio del mes
                                dias_mes = calendar.monthrange(ano, mes)[1]  # Días del mes
                                
                                # Probabilidad de que el pago llegue en este mes específico
                                if dias_predichos <= dias_mes:
                                    # Pago probable en este mes
                                    probabilidad = 1.0
                                elif dias_predichos <= dias_mes + 15:
                                    # Pago parcialmente probable (spillover al siguiente mes)
                                    probabilidad = 0.6
                                elif dias_predichos <= dias_mes + 30:
                                    # Pago menos probable
                                    probabilidad = 0.3
                                else:
                                    # Pago muy improbable en este mes
                                    probabilidad = 0.1
                                
                                # Agregar variabilidad mensual realista
                                # Factores estacionales (algunos meses son mejores que otros)
                                factores_estacionales = {
                                    1: 0.9,   # Enero - inicio lento
                                    2: 0.95,  # Febrero - repunte
                                    3: 1.1,   # Marzo - fuerte
                                    4: 1.05,  # Abril - bueno
                                    5: 0.8,   # Mayo - bajada
                                    6: 0.7,   # Junio - mínimo
                                    7: 0.85,  # Julio - recuperación gradual
                                    8: 0.9,   # Agosto - mejora
                                    9: 1.0,   # Septiembre - normal
                                    10: 1.1,  # Octubre - fuerte
                                    11: 1.05, # Noviembre - bueno
                                    12: 0.6   # Diciembre - mínimo navideño
                                }
                                
                                factor_estacional = factores_estacionales.get(mes, 1.0)
                                
                                # Calcular valor final considerando probabilidad y estacionalidad
                                valor_esperado = valor_promedio * probabilidad * factor_estacional
                                
                                # Solo incluir si el valor esperado es significativo
                                if valor_esperado >= valor_promedio * 0.1:  # Al menos 10% del valor
                                    valor_prediccion_mes += valor_esperado
                                    cantidad_predicciones += 1
                                    
                        except Exception:
                            continue  # Saltar errores individuales
                    
                    if cantidad_predicciones > 0:
                        fuente_datos = "predicciones_ml"
                        # Agregar un factor de ajuste general basado en el mes
                        # (los meses futuros tienen más incertidumbre)
                        meses_desde_hoy = max(0, mes - fecha_actual.month)
                        factor_incertidumbre = max(0.7, 1.0 - (meses_desde_hoy * 0.05))
                        valor_prediccion_mes *= factor_incertidumbre
                        
                except Exception as e:
                    logger.warning(f"Error en predicciones para {mes}/{ano}: {e}")
            
            # 3. DETERMINAR VALOR FINAL DEL MES
            if tiene_datos_reales:
                valor_final_mes = valor_real_mes
                fuente_datos = "datos_reales"
                meses_con_datos += 1
            elif valor_prediccion_mes > 0:
                valor_final_mes = valor_prediccion_mes
                fuente_datos = "predicciones_ml"
                meses_proyectados += 1
            else:
                valor_final_mes = 0
                fuente_datos = "sin_datos"
            
            # 4. AGREGAR AL RESULTADO
            valores_mensuales[f"mes_{mes:02d}"] = {
                "mes": mes,
                "nombre": mes_nombre,
                "valor_estimado": round(valor_final_mes, 2),
                "fuente": fuente_datos,
                "detalles": {
                    "valor_ventas_reales": valor_ventas_real,
                    "valor_cobrado_real": valor_cobrado_real,
                    "valor_predicciones": valor_prediccion_mes,
                    "cantidad_ventas": cantidad_ventas_real,
                    "cantidad_pagos": cantidad_pagos_real,
                    "cantidad_predicciones": cantidad_predicciones
                }
            }
            
            total_anual += valor_final_mes
        
        # 5. RESPUESTA FINAL
        return {
            "ano": ano,
            "fecha_consulta": datetime.now().isoformat(),
            "resumen_anual": {
                "valor_total_estimado": round(total_anual, 2),
                "promedio_mensual": round(total_anual / 12, 2),
                "meses_con_datos_reales": meses_con_datos,
                "meses_proyectados": meses_proyectados,
                "meses_sin_datos": 12 - meses_con_datos - meses_proyectados,
                "porcentaje_datos_reales": round((meses_con_datos / 12) * 100, 2),
                "porcentaje_proyectado": round((meses_proyectados / 12) * 100, 2)
            },
            "valores_mensuales": valores_mensuales,
            "configuracion": {
                "incluir_predicciones": incluir_predicciones,
                "modelo_ml_disponible": modelo_ml.esta_disponible(),
                "filtros_aplicados": ["ADI%", "OTR%", "SPD%", "solo_clientes_pagados_completos"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error en proyección anual simplificada: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/debug/abril_2025_como_companero")
def debug_abril_2025_como_companero(db: Session = Depends(get_db)):
    """
    Replica EXACTAMENTE la lógica del compañero para abril 2025
    """
    try:
        # LÓGICA EXACTA DEL COMPAÑERO:
        # 1. FechaInicio en abril 2025 (NO fecha de facturación)
        # 2. Excluir códigos ADI, OTR, SPD
        # 3. Estado más reciente = 1
        # 4. Factura con estado 3 y pagado > 0
        # 5. Suma de pagos >= valor de la comercialización
        
        comercializaciones_abril = db.query(models.Comercializacion).filter(
            func.extract('year', models.Comercializacion.FechaInicio) == 2025,
            func.extract('month', models.Comercializacion.FechaInicio) == 4,
            ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
            ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
            ~models.Comercializacion.CodigoCotizacion.like('SPD%')
        ).all()
        
        clientes_unicos = set()
        comers_unicas = set()
        total_valor_ventas = 0
        total_valor_pagado = 0
        
        detalles_validacion = []
        
        for comercializacion in comercializaciones_abril:
            # Validar usando la función exacta del compañero
            if validar_cliente_pagado_completo(comercializacion.id, db):
                clientes_unicos.add(comercializacion.ClienteId)
                comers_unicas.add(comercializacion.id)
                
                # Calcular valor de ventas
                valor_venta = comercializacion.ValorVenta or 0
                total_valor_ventas += valor_venta
                
                # Calcular suma de pagos
                facturas = db.query(models.Factura).filter(
                    models.Factura.idComercializacion == comercializacion.id
                ).all()
                
                suma_pagos_com = sum(f.Pagado for f in facturas if f.Pagado and f.Pagado > 0)
                total_valor_pagado += suma_pagos_com
                
                # Agregar detalle para los primeros 10
                if len(detalles_validacion) < 10:
                    detalles_validacion.append({
                        "comercializacion_id": comercializacion.id,
                        "cliente_id": comercializacion.ClienteId,
                        "cliente": comercializacion.Cliente,
                        "codigo_cotizacion": comercializacion.CodigoCotizacion,
                        "fecha_inicio": comercializacion.FechaInicio.isoformat() if comercializacion.FechaInicio else None,
                        "valor_venta": valor_venta,
                        "suma_pagos": suma_pagos_com,
                        "diferencia": suma_pagos_com - valor_venta
                    })
        
        return {
            "metodo": "logica_exacta_companero",
            "criterios": [
                "FechaInicio en abril 2025 (NO fecha facturación)",
                "Excluir códigos ADI, OTR, SPD", 
                "Estado más reciente = 1 (terminada)",
                "Factura con estado 3 y pagado > 0",
                "Suma total pagos >= valor venta"
            ],
            "resultados": {
                "comercializaciones_candidatas": len(comercializaciones_abril),
                "comercializaciones_pagadas_completas": len(comers_unicas),
                "clientes_unicos_pagados": len(clientes_unicos),
                "total_valor_ventas": total_valor_ventas,
                "total_valor_pagado": total_valor_pagado,
                "diferencia": total_valor_pagado - total_valor_ventas
            },
            "sample_detalles": detalles_validacion,
            "fecha_analisis": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error replicando lógica del compañero: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
        
# ===== ENDPOINT SIMPLIFICADO DE PREDICCIÓN POR ID CLIENTE =====

@app.get("/predecir_cliente/{cliente_id}")
def predecir_dias_pago_cliente(cliente_id: int, db: Session = Depends(get_db)):
    """
    Predice los días de pago para un cliente usando solo su ID.
    Utiliza datos históricos del cliente para generar predicción automática.
    
    Args:
        cliente_id: ID del cliente en la base de datos
    
    Returns:
        Predicción de días de pago con información del cliente
    """
    if not modelo_ml.esta_disponible():
        raise HTTPException(status_code=503, detail="Modelo de ML no disponible")
    
    try:
        # 1. OBTENER INFORMACIÓN DEL CLIENTE
        cliente_info = db.query(models.Comercializacion).filter(
            models.Comercializacion.ClienteId == cliente_id,
            models.Comercializacion.Cliente.isnot(None),
            ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
            ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
            ~models.Comercializacion.CodigoCotizacion.like('SPD%')
        ).first()
        
        if not cliente_info:
            raise HTTPException(status_code=404, detail=f"Cliente con ID {cliente_id} no encontrado")
        
        # 2. CALCULAR ESTADÍSTICAS HISTÓRICAS DEL CLIENTE
        comercializaciones_cliente = db.query(models.Comercializacion).filter(
            models.Comercializacion.ClienteId == cliente_id,
            models.Comercializacion.ValorVenta.isnot(None),
            ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
            ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
            ~models.Comercializacion.CodigoCotizacion.like('SPD%')
        ).all()
        
        if not comercializaciones_cliente:
            raise HTTPException(status_code=404, detail=f"No se encontraron comercializaciones válidas para el cliente {cliente_id}")
        
        # Calcular valor promedio
        valores_venta = [c.ValorVenta for c in comercializaciones_cliente if c.ValorVenta and c.ValorVenta > 0]
        valor_promedio = sum(valores_venta) / len(valores_venta) if valores_venta else 500000
        
        # Determinar si es cliente SENCE (mayoría de sus comercializaciones)
        total_comercializaciones = len(comercializaciones_cliente)
        comercializaciones_sence = sum(1 for c in comercializaciones_cliente if c.EsSENCE)
        es_cliente_sence = comercializaciones_sence > (total_comercializaciones / 2)
        
        # Determinar correo basado en tipo de cliente
        correo_frecuente = "sence@insecap.cl" if es_cliente_sence else "comercial@insecap.cl"
        
        # Calcular número promedio de facturas
        facturas_por_comercializacion = []
        for c in comercializaciones_cliente:
            facturas_count = db.query(models.Factura).filter(
                models.Factura.idComercializacion == c.id
            ).count()
            facturas_por_comercializacion.append(facturas_count)
        
        cantidad_facturas_promedio = int(sum(facturas_por_comercializacion) / len(facturas_por_comercializacion)) if facturas_por_comercializacion else 1
        cantidad_facturas_promedio = max(1, min(cantidad_facturas_promedio, 5))  # Entre 1 y 5
        
        # 3. PREPARAR DATOS PARA PREDICCIÓN
        mes_actual = datetime.now().month
        datos_prediccion = {
            "cliente": cliente_info.Cliente,
            "correo_creador": correo_frecuente,
            "valor_venta": valor_promedio,
            "es_sence": es_cliente_sence,
            "mes_facturacion": mes_actual,
            "cantidad_facturas": cantidad_facturas_promedio
        }
        
        # 4. REALIZAR PREDICCIÓN
        resultado_prediccion = modelo_ml.predecir_dias_pago(datos_prediccion)
        
        # 5. ENRIQUECER RESPUESTA CON INFORMACIÓN DEL CLIENTE
        respuesta_completa = {
            "cliente_id": cliente_id,
            "cliente_nombre": cliente_info.Cliente,
            "prediccion": {
                "dias_predichos": resultado_prediccion["dias_predichos"],
                "nivel_riesgo": resultado_prediccion["nivel_riesgo"],
                "codigo_riesgo": resultado_prediccion["codigo_riesgo"],
                "descripcion_riesgo": resultado_prediccion["descripcion_riesgo"],
                "accion_recomendada": resultado_prediccion["accion_recomendada"],
                "confianza": resultado_prediccion["confianza"],
                "se_paga_mismo_mes": resultado_prediccion["se_paga_mismo_mes"],
                "explicacion_mes": resultado_prediccion["explicacion_mes"]
            },
            "datos_utilizados": {
                "valor_venta_promedio": round(valor_promedio, 2),
                "es_cliente_sence": es_cliente_sence,
                "cantidad_facturas_promedio": cantidad_facturas_promedio,
                "mes_prediccion": mes_actual,
                "correo_asignado": correo_frecuente
            },
            "estadisticas_cliente": {
                "total_comercializaciones": total_comercializaciones,
                "comercializaciones_sence": comercializaciones_sence,
                "porcentaje_sence": round((comercializaciones_sence / total_comercializaciones) * 100, 2),
                "valor_minimo": min(valores_venta) if valores_venta else 0,
                "valor_maximo": max(valores_venta) if valores_venta else 0,
                "valor_promedio": round(valor_promedio, 2)
            },
            "fecha_prediccion": datetime.now().isoformat(),
            "modelo_version": resultado_prediccion.get("modelo_version", "Híbrido v2.0")
        }
        
        return respuesta_completa
        
    except HTTPException:
        raise  # Re-lanzar HTTPExceptions
    except Exception as e:
        logger.error(f"Error en predicción por cliente ID {cliente_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/predecir_cliente/{cliente_id}/resumen")
def predecir_dias_pago_cliente_resumen(cliente_id: int, db: Session = Depends(get_db)):
    """
    Versión resumida de predicción por ID de cliente (solo lo esencial).
    
    Args:
        cliente_id: ID del cliente en la base de datos
    
    Returns:
        Predicción resumida con información básica
    """
    if not modelo_ml.esta_disponible():
        raise HTTPException(status_code=503, detail="Modelo de ML no disponible")
    
    try:
        # Obtener información básica del cliente
        cliente_info = db.query(models.Comercializacion).filter(
            models.Comercializacion.ClienteId == cliente_id,
            models.Comercializacion.Cliente.isnot(None),
            ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
            ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
            ~models.Comercializacion.CodigoCotizacion.like('SPD%')
        ).first()
        
        if not cliente_info:
            raise HTTPException(status_code=404, detail=f"Cliente con ID {cliente_id} no encontrado")
        
        # Obtener comercializaciones para calcular promedio
        comercializaciones = db.query(models.Comercializacion.ValorVenta, models.Comercializacion.EsSENCE).filter(
            models.Comercializacion.ClienteId == cliente_id,
            models.Comercializacion.ValorVenta.isnot(None),
            ~models.Comercializacion.CodigoCotizacion.like('ADI%'),
            ~models.Comercializacion.CodigoCotizacion.like('OTR%'),
            ~models.Comercializacion.CodigoCotizacion.like('SPD%')
        ).all()
        
        if not comercializaciones:
            raise HTTPException(status_code=404, detail="No hay datos suficientes para predicción")
        
        # Calcular datos básicos
        valores = [c[0] for c in comercializaciones if c[0] and c[0] > 0]
        valor_promedio = sum(valores) / len(valores) if valores else 500000
        
        sence_count = sum(1 for c in comercializaciones if c[1])
        es_sence = sence_count > (len(comercializaciones) / 2)
        
        # Calcular cantidad de facturas promedio realista
        # Para valores pequeños, usar 1 factura (más común)
        if valor_promedio <= 200000:
            cantidad_facturas = 1
        elif valor_promedio <= 500000:
            cantidad_facturas = 2
        else:
            cantidad_facturas = 3
        
        # Preparar datos mínimos para predicción
        datos_prediccion = {
            "cliente": cliente_info.Cliente,
            "correo_creador": "prediccion@insecap.cl",
            "valor_venta": valor_promedio,
            "es_sence": es_sence,
            "mes_facturacion": datetime.now().month,
            "cantidad_facturas": cantidad_facturas
        }
        
        # Realizar predicción
        resultado = modelo_ml.predecir_dias_pago(datos_prediccion)
        
        # Respuesta simplificada
        return {
            "cliente_id": cliente_id,
            "cliente": cliente_info.Cliente,
            "dias_predichos": resultado["dias_predichos"],
            "nivel_riesgo": resultado["codigo_riesgo"],
            "se_paga_este_mes": resultado["se_paga_mismo_mes"],
            "confianza": round(resultado["confianza"], 3),
            "tipo_cliente": "SENCE" if es_sence else "COMERCIAL",
            "valor_promedio": round(valor_promedio, 2),
            "fecha_prediccion": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción resumida cliente {cliente_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
