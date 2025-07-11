from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, DateTime, Text, Boolean
from database import Base

# Comercializaciones
class Comercializacion(Base):
    __tablename__ = "comercializaciones"

    id = Column(Integer, primary_key=True, index=True)
    CodigoCotizacion = Column(String, index=True)
    Cliente = Column(String)
    ClienteId = Column(Integer)
    LiderComercial = Column(String)
    FechaInicio = Column(Date)
    ValorVenta = Column(Float)
    ValorCotizacion = Column(Float)
    NumeroEstados = Column(Integer)
    EsSENCE = Column(Integer)  # 0 o 1


# Estados
class Estado(Base):
    __tablename__ = "estados"

    id = Column(Integer, primary_key=True, index=True)
    idComercializacion = Column(Integer, ForeignKey("comercializaciones.id"))
    EstadoComercializacion = Column(Integer)
    Fecha = Column(Date)


# Facturas
class Factura(Base):
    __tablename__ = "facturas"

    id = Column(Integer, primary_key=True, index=True)
    idComercializacion = Column(Integer, ForeignKey("comercializaciones.id"))
    CodigoCotizacion = Column(String)
    NumeroFactura = Column(String)
    FechaFacturacion = Column(Date)
    NumeroEstadosFactura = Column(Integer)
    EstadoFactura = Column(Integer)
    FechaEstado = Column(Date)
    Pagado = Column(Float)
    Usuario = Column(String)
    Observacion = Column(String)


# Nueva tabla para predicciones almacenadas
class PrediccionAlmacenada(Base):
    __tablename__ = "predicciones_almacenadas"

    id = Column(Integer, primary_key=True, index=True)
    cliente_id = Column(Integer, index=True)
    cliente_nombre = Column(String, index=True)
    lider_comercial = Column(String, index=True)
    
    # Datos de entrada de la predicción
    valor_venta = Column(Float)
    es_sence = Column(Boolean)
    mes_prediccion = Column(Integer)
    ano_prediccion = Column(Integer)
    
    # Resultados de la predicción
    dias_predichos = Column(Integer)
    fecha_pago_estimada = Column(Date)
    nivel_riesgo = Column(String)
    codigo_riesgo = Column(String)
    confianza = Column(Float)
    
    # Metadatos
    fecha_creacion = Column(DateTime)
    modelo_version = Column(String)
    activa = Column(Boolean, default=True)  # Para marcar predicciones obsoletas
    notas = Column(Text)


# Análisis históricos por cliente y líder comercial
class AnalisisHistorico(Base):
    __tablename__ = "analisis_historicos"

    id = Column(Integer, primary_key=True, index=True)
    
    # Identificadores
    cliente_id = Column(Integer, index=True)
    cliente_nombre = Column(String, index=True)
    lider_comercial = Column(String, index=True)
    
    # Métricas de tiempo (deltas)
    delta_x_promedio = Column(Float)  # Inicio → Terminado
    delta_y_promedio = Column(Float)  # Terminado → Primera Factura
    delta_z_promedio = Column(Float)  # Primera Factura → Último Pago
    delta_g_promedio = Column(Float)  # Inicio → Último Pago (total)
    
    # Estadísticas adicionales
    total_transacciones = Column(Integer)
    valor_promedio_venta = Column(Float)
    porcentaje_sence = Column(Float)
    fecha_ultima_actualizacion = Column(DateTime)
    periodo_analisis_inicio = Column(Date)
    periodo_analisis_fin = Column(Date)

# ===== MODELOS PYDANTIC PARA PREDICCIONES ML =====

from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class VentaInput(BaseModel):
    """Modelo de entrada para predicciones de ventas"""
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
    """Modelo de respuesta para predicciones"""
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
    """Modelo de respuesta para estadísticas de ML"""
    total_predicciones: int
    promedio_dias: float
    distribuciones_riesgo: Dict[str, int]
    modelo_info: Dict[str, Any]
    timestamp: datetime

class PrediccionAlmacenadaInput(BaseModel):
    """Modelo para crear predicciones almacenadas"""
    cliente_id: int
    cliente_nombre: str
    lider_comercial: str
    valor_venta: float
    es_sence: bool
    mes_prediccion: int
    ano_prediccion: int
    notas: Optional[str] = None

class PrediccionAlmacenadaResponse(BaseModel):
    """Modelo de respuesta para predicciones almacenadas"""
    id: int
    cliente_id: int
    cliente_nombre: str
    lider_comercial: str
    valor_venta: float
    es_sence: bool
    mes_prediccion: int
    ano_prediccion: int
    dias_predichos: int
    fecha_pago_estimada: datetime
    nivel_riesgo: str
    codigo_riesgo: str
    confianza: float
    fecha_creacion: datetime
    modelo_version: str
    activa: bool
    notas: Optional[str]

class AnalisisHistoricoResponse(BaseModel):
    """Modelo de respuesta para análisis históricos"""
    cliente_id: int
    cliente_nombre: str
    lider_comercial: str
    delta_x_promedio: float
    delta_y_promedio: float
    delta_z_promedio: float
    delta_g_promedio: float
    total_transacciones: int
    valor_promedio_venta: float
    porcentaje_sence: float
    fecha_ultima_actualizacion: datetime
    periodo_analisis_inicio: datetime
    periodo_analisis_fin: datetime

class GenerarPrediccionesMasivasInput(BaseModel):
    """Modelo para generar predicciones masivas"""
    ano_inicio: int
    ano_fin: int
    incluir_clientes_activos_solamente: bool = True
    sobrescribir_existentes: bool = False
    limite_clientes: Optional[int] = None