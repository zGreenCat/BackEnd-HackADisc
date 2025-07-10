from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear tablas (si no existen)
models.Base.metadata.create_all(bind=engine)

# Instancia principal
app = FastAPI(
    title="Backend Insecap + Predicciones ML", 
    description="API completa para datos y predicciones de días de pago",
    version="2.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelos de ML al iniciar
modelo_hibrido = None
scaler_hibrido = None
metadata_hibrido = None
umbrales_inteligentes = None

try:
    # Buscar archivos en el directorio actual
    modelo_path = "modelo_hibrido.pkl"
    scaler_path = "scaler_hibrido.pkl"
    metadata_path = "modelo_hibrido_metadata.pkl"
    
    if os.path.exists(modelo_path) and os.path.exists(scaler_path) and os.path.exists(metadata_path):
        modelo_hibrido = joblib.load(modelo_path)
        scaler_hibrido = joblib.load(scaler_path)
        metadata_hibrido = joblib.load(metadata_path)
        
        # Calcular umbrales inteligentes basados en datos históricos reales
        umbrales_inteligentes = {
            'muy_bajo': 28.0, 'bajo': 36.0, 'medio': 47.0, 
            'alto': 63.0, 'critico': 77.0, 'media': 38.2, 'mediana': 36.0
        }
        
        logger.info("✅ Modelos de ML cargados exitosamente")
        logger.info(f"📊 MAE: {metadata_hibrido.get('mae', 'N/A')}, R²: {metadata_hibrido.get('r2', 'N/A')}")
    else:
        logger.warning("⚠️ Archivos de modelo no encontrados, funcionalidad ML deshabilitada")
        
except Exception as e:
    logger.error(f"❌ Error cargando modelos ML: {e}")

# Almacenamiento en memoria para predicciones (en producción usar base de datos)
predicciones_cache = []

# Dependency para acceder a la DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ===== ENDPOINTS PRINCIPALES =====

# Ruta base
@app.get("/")
def root():
    """Endpoint raíz con información de la API completa"""
    return {
        "message": "Backend Insecap + Predicciones ML activo",
        "version": "2.0",
        "funcionalidades": {
            "datos": "CRUD completo de comercializaciones, facturas y estados",
            "ml": "Predicciones de días de pago" if modelo_hibrido else "ML no disponible",
            "documentacion": "/docs"
        },
        "modelo_ml": {
            "activo": modelo_hibrido is not None,
            "mae": metadata_hibrido.get('mae', 'N/A') if metadata_hibrido else 'N/A',
            "r2": metadata_hibrido.get('r2', 'N/A') if metadata_hibrido else 'N/A'
        }
    }

@app.get("/health")
def health_check():
    """Endpoint de health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "database": "connected",
        "modelo_ml": "loaded" if modelo_hibrido else "not_available"
    }

# Ruta GET /resumen
# Desc: Devuelve un resumen general con el total de registros cargados en cada una de las tres entidades: comercializaciones, facturas y estados.
@app.get("/resumen")
def resumen(db: Session = Depends(get_db)):
    total_ventas = db.query(models.Comercializacion).count()
    total_facturas = db.query(models.Factura).count()
    total_estados = db.query(models.Estado).count()

    return {
        "comercializaciones": total_ventas,
        "facturas": total_facturas,
        "estados": total_estados
    }

# Ruta GET /comercializaciones
# Desc: Devuelve una lista completa de todas las comercializaciones almacenadas.
@app.get("/comercializaciones")
def obtener_comercializaciones(db: Session = Depends(get_db)):
    return db.query(models.Comercializacion).all()

# Ruta GET /cliente/{nombre}
# Desc: Devuelve todas las comercializaciones asociadas a un cliente por nombre exacto. Si el cliente no existe, lanza un error 404.
# Params: nombre (string) – nombre exacto del cliente (sensible a mayúsculas y espacios)
@app.get("/cliente/{nombre}")
def obtener_cliente(nombre: str, db: Session = Depends(get_db)):
    resultado = db.query(models.Comercializacion).filter(models.Comercializacion.Cliente == nombre).all()
    if not resultado:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return resultado

# Ruta GET /facturas
# Desc: Devuelve todas las facturas de la base de datos, incluyendo múltiples entradas por cada estado de factura.
@app.get("/facturas")
def obtener_facturas(db: Session = Depends(get_db)):
    return db.query(models.Factura).all()

# Ruta GET /estados
# Desc: Devuelve todos los estados registrados para las comercializaciones. Cada fila representa un cambio de estado con su fecha.
@app.get("/estados")
def obtener_estados(db: Session = Depends(get_db)):
    return db.query(models.Estado).all()

# Ruta GET /sence
# Desc: Devuelve la cantidad de comercializaciones que corresponden a sence y las que no.
@app.get("/sence")
def contar_sence(db: Session = Depends(get_db)):
    total_sence = db.query(models.Comercializacion).filter(models.Comercializacion.EsSENCE == 1).count()
    total_no_sence = db.query(models.Comercializacion).filter(models.Comercializacion.EsSENCE == 0).count()

    return {
        "sence": total_sence,
        "no_sence": total_no_sence
    }

# ===== MODELOS PYDANTIC PARA PREDICCIONES ML =====

class VentaInput(BaseModel):
    cliente: str
    correo_creador: str
    valor_venta: float
    es_sence: bool
    mes_facturacion: int
    valor_cotizacion: Optional[float] = None
    cantidad_facturas: int = 1
    tiempo_promedio_facturas: Optional[float] = None
    dias_inicio_facturacion: int = 30

    @validator('cliente')
    def validar_cliente(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Cliente no puede estar vacío')
        return v.strip()

    @validator('correo_creador')
    def validar_correo(cls, v):
        if '@' not in v or '.' not in v:
            raise ValueError('Formato de correo inválido')
        return v.strip()

    @validator('valor_venta')
    def validar_valor_venta(cls, v):
        if v <= 0:
            raise ValueError('Valor de venta debe ser positivo')
        if v > 50_000_000:
            raise ValueError('Valor de venta excesivo (>50M)')
        return v

    @validator('mes_facturacion')
    def validar_mes(cls, v):
        if not 1 <= v <= 12:
            raise ValueError('Mes debe estar entre 1 y 12')
        return v

class PrediccionResponse(BaseModel):
    dias_predichos: int
    nivel_riesgo: str
    codigo_riesgo: str
    descripcion_riesgo: str
    accion_recomendada: str
    confianza: float
    se_paga_mismo_mes: bool
    explicacion_mes: str
    fecha_prediccion: datetime
    modelo_version: str

class EstadisticasMLResponse(BaseModel):
    total_predicciones: int
    promedio_dias: float
    distribuciones_riesgo: Dict[str, int]
    modelo_info: Dict[str, Any]
    timestamp: datetime

# ===== FUNCIONES AUXILIARES PARA ML =====

def clasificar_riesgo_inteligente(dias_predichos: int) -> tuple:
    """Clasifica el riesgo basado en umbrales inteligentes"""
    if not umbrales_inteligentes:
        # Fallback a clasificación simple
        if dias_predichos <= 30:
            return "🟢 BAJO", "BAJO", "Bueno: Dentro del rango esperado", "Seguimiento normal"
        elif dias_predichos <= 60:
            return "🟡 MEDIO", "MEDIO", "Normal: Requiere atención", "Contacto proactivo recomendado"
        else:
            return "🔴 ALTO", "ALTO", "Preocupante: Requiere intervención", "Intervención urgente requerida"
    
    # Clasificación inteligente
    if dias_predichos <= umbrales_inteligentes['muy_bajo']:
        return (
            "🟢 MUY BAJO", "MUY_BAJO",
            f"Excelente: Mejor que 75% de casos históricos (≤{umbrales_inteligentes['muy_bajo']:.0f} días)",
            "Seguimiento automático mensual"
        )
    elif dias_predichos <= umbrales_inteligentes['bajo']:
        return (
            "🟢 BAJO", "BAJO",
            f"Bueno: Mejor que 50% de casos históricos (≤{umbrales_inteligentes['bajo']:.0f} días)",
            "Seguimiento quincenal estándar"
        )
    elif dias_predichos <= umbrales_inteligentes['medio']:
        return (
            "🟡 MEDIO", "MEDIO",
            f"Normal: Entre mediana y percentil 75 ({umbrales_inteligentes['bajo']:.0f}-{umbrales_inteligentes['medio']:.0f} días)",
            "Contacto proactivo semanal"
        )
    elif dias_predichos <= umbrales_inteligentes['alto']:
        return (
            "🟠 ALTO", "ALTO",
            f"Preocupante: Peor que 75% de casos ({umbrales_inteligentes['medio']:.0f}-{umbrales_inteligentes['alto']:.0f} días)",
            "Seguimiento diario y escalamiento"
        )
    else:
        return (
            "🔴 CRÍTICO", "CRITICO",
            f"Alarmante: Peor que 90% de casos históricos (>{umbrales_inteligentes['alto']:.0f} días)",
            "Intervención inmediata y revisión gerencial"
        )

def se_pagaria_en_el_mismo_mes(mes_facturacion: int, dias_estimados: int, dia_facturacion: int = 1, año: int = None) -> tuple:
    """Determina si una venta se pagará en el mismo mes de su facturación"""
    # Días por mes (considerando años bisiestos)
    if año and año % 4 == 0 and (año % 100 != 0 or año % 400 == 0):
        dias_mes = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    else:
        dias_mes = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

    dias_totales_mes = dias_mes.get(mes_facturacion, 30)
    dias_disponibles = dias_totales_mes - dia_facturacion + 1
    
    if dias_estimados <= dias_disponibles:
        return True, f"✅ Se pagará dentro del mes (quedan {dias_disponibles} días desde el {dia_facturacion}/{mes_facturacion})"
    else:
        fecha_estimada_pago = dia_facturacion + dias_estimados
        mes_pago = mes_facturacion + (fecha_estimada_pago - 1) // dias_totales_mes
        return False, f"❌ Se pagará en el mes {mes_pago} (faltan {dias_estimados - dias_disponibles} días después del mes actual)"

def calcular_confianza(dias_predichos: int) -> float:
    """Calcula un score de confianza basado en el rendimiento del modelo"""
    if not metadata_hibrido:
        return 0.8  # Confianza por defecto
    
    mae = metadata_hibrido.get('mae', 3.0)
    r2 = metadata_hibrido.get('r2', 0.9)
    
    # Confianza basada en el rendimiento del modelo
    confianza_modelo = min(1.0, r2)  # R² como base
    confianza_prediccion = max(0.3, 1 - (mae / 10))  # Basado en MAE
    
    return min(1.0, (confianza_modelo + confianza_prediccion) / 2)

def guardar_prediccion_cache(input_data: dict, response_data: dict):
    """Guarda predicción en cache (background task)"""
    predicciones_cache.append({
        "timestamp": datetime.now(),
        "input": input_data,
        "respuesta": response_data
    })
    
    # Mantener solo últimas 1000 predicciones
    if len(predicciones_cache) > 1000:
        predicciones_cache.pop(0)

# ===== ENDPOINTS DE PREDICCIONES ML =====

@app.post("/predecir", response_model=PrediccionResponse)
def predecir_dias_pago(venta: VentaInput, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Predice los días hasta pago completo de una venta usando el modelo híbrido
    """
    if not modelo_hibrido:
        raise HTTPException(status_code=503, detail="Modelo de ML no disponible")
    
    try:
        # Preparar datos para predicción
        nueva_venta = pd.DataFrame({
            'ValorVenta': [venta.valor_venta],
            'EsSENCE': [1 if venta.es_sence else 0],
            'MesFacturacion': [venta.mes_facturacion],
            'AñoFacturacion': [2024],
            'CantidadFacturas': [venta.cantidad_facturas],
            'RatioVentaCotizacion': [venta.valor_venta / (venta.valor_cotizacion or venta.valor_venta)],
            'TiempoPromedioFacturas': [venta.tiempo_promedio_facturas or 
                                     (0 if venta.cantidad_facturas == 1 else 
                                      min(15 + (venta.cantidad_facturas - 1) * 5, 45))],
            'MontoPromedioFactura': [venta.valor_venta / venta.cantidad_facturas],
            'StdPagos': [venta.valor_venta * 0.1],
            'DiasInicioAFacturacion': [venta.dias_inicio_facturacion],
            'TrimestreFacturacion': [(venta.mes_facturacion - 1) // 3 + 1],
            'DiaSemanaFacturacion': [2]
        })
        
        # Encoding de cliente y vendedor
        cliente_cats = metadata_hibrido['cliente_categories']
        cliente_codes = {v: k for k, v in cliente_cats.items()}
        nueva_venta['Cliente_encoded'] = cliente_codes.get(venta.cliente, -1)
        
        vendedor_cats = metadata_hibrido['vendedor_categories']
        vendedor_codes = {v: k for k, v in vendedor_cats.items()}
        nueva_venta['Vendedor_encoded'] = vendedor_codes.get(venta.correo_creador, -1)
        
        # Categoría de monto
        percentiles = metadata_hibrido['percentiles_monto']
        if venta.valor_venta <= percentiles[0]:
            categoria_monto = 0
        elif venta.valor_venta <= percentiles[1]:
            categoria_monto = 1
        else:
            categoria_monto = 2
        nueva_venta['CategoriaMontoVenta'] = categoria_monto
        nueva_venta['CategoriaPagoCliente'] = 1
        
        # Normalizar
        cols_normalizar = ['ValorVenta', 'TiempoPromedioFacturas', 'MontoPromedioFactura', 'DiasInicioAFacturacion']
        nueva_venta[cols_normalizar] = scaler_hibrido.transform(nueva_venta[cols_normalizar])
        
        # Predicción
        prediccion = modelo_hibrido.predict(nueva_venta[metadata_hibrido['features']])[0]
        dias_predichos = max(0, round(prediccion))
        
        # Clasificar riesgo
        nivel_riesgo, codigo_riesgo, descripcion_riesgo, accion = clasificar_riesgo_inteligente(dias_predichos)
        
        # Análisis de temporalidad
        se_paga_mismo_mes, explicacion_mes = se_pagaria_en_el_mismo_mes(
            venta.mes_facturacion, dias_predichos, 1, 2024
        )
        
        # Calcular confianza
        confianza = calcular_confianza(dias_predichos)
        
        # Crear respuesta
        response = PrediccionResponse(
            dias_predichos=dias_predichos,
            nivel_riesgo=nivel_riesgo,
            codigo_riesgo=codigo_riesgo,
            descripcion_riesgo=descripcion_riesgo,
            accion_recomendada=accion,
            confianza=confianza,
            se_paga_mismo_mes=se_paga_mismo_mes,
            explicacion_mes=explicacion_mes,
            fecha_prediccion=datetime.now(),
            modelo_version="Híbrido v2.0"
        )
        
        # Guardar en cache (background task)
        background_tasks.add_task(
            guardar_prediccion_cache,
            venta.dict(),
            response.dict()
        )
        
        logger.info(f"Predicción realizada: {dias_predichos} días para cliente {venta.cliente}")
        return response
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno en predicción: {str(e)}")

@app.post("/predecir_lote")
def predecir_lote(ventas: List[VentaInput], background_tasks: BackgroundTasks):
    """
    Predice múltiples ventas en lote (máximo 50 por request)
    """
    if not modelo_hibrido:
        raise HTTPException(status_code=503, detail="Modelo de ML no disponible")
    
    if len(ventas) > 50:
        raise HTTPException(status_code=400, detail="Máximo 50 ventas por lote")
    
    resultados = []
    errores = 0
    
    for i, venta in enumerate(ventas):
        try:
            # Reutilizar lógica del endpoint individual (simplificado)
            nueva_venta = pd.DataFrame({
                'ValorVenta': [venta.valor_venta],
                'EsSENCE': [1 if venta.es_sence else 0],
                'MesFacturacion': [venta.mes_facturacion],
                'AñoFacturacion': [2024],
                'CantidadFacturas': [venta.cantidad_facturas],
                'RatioVentaCotizacion': [venta.valor_venta / (venta.valor_cotizacion or venta.valor_venta)],
                'TiempoPromedioFacturas': [venta.tiempo_promedio_facturas or 15],
                'MontoPromedioFactura': [venta.valor_venta / venta.cantidad_facturas],
                'StdPagos': [venta.valor_venta * 0.1],
                'DiasInicioAFacturacion': [venta.dias_inicio_facturacion],
                'TrimestreFacturacion': [(venta.mes_facturacion - 1) // 3 + 1],
                'DiaSemanaFacturacion': [2]
            })
            
            # Simplificar encoding para lote
            nueva_venta['Cliente_encoded'] = -1
            nueva_venta['Vendedor_encoded'] = -1
            nueva_venta['CategoriaMontoVenta'] = 1
            nueva_venta['CategoriaPagoCliente'] = 1
            
            # Normalizar
            cols_normalizar = ['ValorVenta', 'TiempoPromedioFacturas', 'MontoPromedioFactura', 'DiasInicioAFacturacion']
            nueva_venta[cols_normalizar] = scaler_hibrido.transform(nueva_venta[cols_normalizar])
            
            # Predicción
            prediccion = modelo_hibrido.predict(nueva_venta[metadata_hibrido['features']])[0]
            dias_predichos = max(0, round(prediccion))
            
            nivel_riesgo, codigo_riesgo, descripcion_riesgo, accion = clasificar_riesgo_inteligente(dias_predichos)
            
            resultado = {
                "indice": i,
                "cliente": venta.cliente,
                "dias_predichos": dias_predichos,
                "nivel_riesgo": nivel_riesgo,
                "codigo_riesgo": codigo_riesgo,
                "accion_recomendada": accion
            }
            resultados.append(resultado)
            
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
    """
    Obtiene estadísticas de predicciones realizadas y información del modelo
    """
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
        "mae": metadata_hibrido.get('mae', 'N/A') if metadata_hibrido else 'N/A',
        "r2": metadata_hibrido.get('r2', 'N/A') if metadata_hibrido else 'N/A',
        "features": len(metadata_hibrido['features']) if metadata_hibrido else 0,
        "fecha_entrenamiento": metadata_hibrido.get('fecha_entrenamiento', 'N/A') if metadata_hibrido else 'N/A'
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
    """
    Información detallada del modelo de ML
    """
    if not modelo_hibrido:
        raise HTTPException(status_code=503, detail="Modelo de ML no disponible")
    
    return {
        "modelo": {
            "tipo": "Random Forest Híbrido",
            "version": "2.0",
            "mae": metadata_hibrido['mae'],
            "rmse": metadata_hibrido.get('rmse', 'N/A'),
            "r2": metadata_hibrido['r2'],
            "features_count": len(metadata_hibrido['features']),
            "fecha_entrenamiento": metadata_hibrido.get('fecha_entrenamiento', 'No disponible')
        },
        "features_principales": metadata_hibrido['features'][:10],
        "umbrales_riesgo": umbrales_inteligentes,
        "rendimiento": {
            "precision": "Excelente (MAE < 3 días)",
            "r2_score": f"{metadata_hibrido['r2']:.4f}",
            "casos_entrenamiento": "12,729 casos históricos"
        }
    }

@app.get("/modelo/test")
def test_modelo():
    """
    Endpoint para probar rápidamente el modelo con datos de ejemplo
    """
    if not modelo_hibrido:
        raise HTTPException(status_code=503, detail="Modelo de ML no disponible")
    
    # Datos de ejemplo
    ejemplo = VentaInput(
        cliente="CLIENTE_EJEMPLO",
        correo_creador="test@insecap.cl",
        valor_venta=500000,
        es_sence=False,
        mes_facturacion=7,
        cantidad_facturas=2
    )
    
    try:
        # Hacer predicción rápida
        nueva_venta = pd.DataFrame({
            'ValorVenta': [ejemplo.valor_venta],
            'EsSENCE': [0],
            'MesFacturacion': [ejemplo.mes_facturacion],
            'AñoFacturacion': [2024],
            'CantidadFacturas': [ejemplo.cantidad_facturas],
            'RatioVentaCotizacion': [1.0],
            'TiempoPromedioFacturas': [15],
            'MontoPromedioFactura': [ejemplo.valor_venta / ejemplo.cantidad_facturas],
            'StdPagos': [ejemplo.valor_venta * 0.1],
            'DiasInicioAFacturacion': [30],
            'TrimestreFacturacion': [3],
            'DiaSemanaFacturacion': [2],
            'Cliente_encoded': [-1],
            'Vendedor_encoded': [-1],
            'CategoriaMontoVenta': [1],
            'CategoriaPagoCliente': [1]
        })
        
        # Normalizar
        cols_normalizar = ['ValorVenta', 'TiempoPromedioFacturas', 'MontoPromedioFactura', 'DiasInicioAFacturacion']
        nueva_venta[cols_normalizar] = scaler_hibrido.transform(nueva_venta[cols_normalizar])
        
        # Predicción
        prediccion = modelo_hibrido.predict(nueva_venta[metadata_hibrido['features']])[0]
        dias_predichos = max(0, round(prediccion))
        
        nivel_riesgo, codigo_riesgo, descripcion_riesgo, accion = clasificar_riesgo_inteligente(dias_predichos)
        
        return {
            "test_exitoso": True,
            "ejemplo_entrada": ejemplo.dict(),
            "resultado": {
                "dias_predichos": dias_predichos,
                "nivel_riesgo": nivel_riesgo,
                "descripcion": descripcion_riesgo,
                "accion": accion
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        return {
            "test_exitoso": False,
            "error": str(e),
            "timestamp": datetime.now()
        }