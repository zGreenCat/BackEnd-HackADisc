from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models

# Crear tablas (si no existen)
models.Base.metadata.create_all(bind=engine)

# Instancia principal
app = FastAPI(title="Backend Insecap", version="1.0")

# Dependency para acceder a la DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Ruta base
# Desc: Ruta raíz para verificar que la API está activa.
@app.get("/")
def root():
    return {"message": "Backend Insecap activo"}

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