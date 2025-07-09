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