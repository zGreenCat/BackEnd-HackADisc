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
from models import VentaInput, PrediccionResponse, EstadisticasMLResponse
from ml_predictor import modelo_ml
from statistics import mean, stdev
from fastapi import Path

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

# ===== DEPENDENCY INJECTION =====

def get_db():
    """Dependency para acceder a la base de datos"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
    """Devuelve un resumen general con el total de registros cargados"""
    total_ventas = db.query(models.Comercializacion).count()
    total_facturas = db.query(models.Factura).count()
    total_estados = db.query(models.Estado).count()

    return {
        "comercializaciones": total_ventas,
        "facturas": total_facturas,
        "estados": total_estados,
        "total_general": total_ventas + total_facturas + total_estados
    }

@app.get("/")
def obtener_comercializaciones(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    """Devuelve comercializaciones con paginación"""
    comercializaciones = db.query(models.Comercializacion).offset(offset).limit(limit).all()
    total = db.query(models.Comercializacion).count()
    
    return {
        "comercializaciones": comercializaciones,
        "total": total,
        "limit": limit,
        "offset": offset
    }

@app.get("/cliente/{nombre}")
def obtener_cliente(nombre: str, db: Session = Depends(get_db)):
    """Devuelve todas las comercializaciones asociadas a un cliente por nombre exacto"""
    resultado = db.query(models.Comercializacion).filter(models.Comercializacion.Cliente == nombre).all()
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
    """Devuelve la cantidad de comercializaciones SENCE vs no-SENCE"""
    total_sence = db.query(models.Comercializacion).filter(models.Comercializacion.EsSENCE == 1).count()
    total_no_sence = db.query(models.Comercializacion).filter(models.Comercializacion.EsSENCE == 0).count()

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
    """Lista todos los clientes únicos"""
    clientes = db.query(models.Comercializacion.Cliente).distinct().all()
    lista_clientes = [cliente[0] for cliente in clientes if cliente[0]]
    
    return {
        "clientes": sorted(lista_clientes),
        "total_clientes": len(lista_clientes)
    }

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

    # Obtener ventas activas del cliente
    ventas = (
        db.query(models.Comercializacion)
        .filter(models.Comercializacion.ClienteId == cliente_id)
        .filter(models.Comercializacion.EstadoComercializacion.in_([0, 1, 3]))  # En proceso, terminado, SENCE
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

# ===== ENDPOINTS ENTREGABLES =====

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
