from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey
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