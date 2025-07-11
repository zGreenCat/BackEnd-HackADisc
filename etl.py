import pandas as pd
import json
from datetime import datetime
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models

def parse_fecha(fecha_str):
    try:
        return datetime.strptime(fecha_str, "%d/%m/%Y")
    except:
        return None

def cargar_json_a_db(json_path="json_completo.json"):
    # Crear todas las tablas si no existen
    print("🔧 Creando tablas en la base de datos...")
    models.Base.metadata.create_all(bind=engine)
    print("✅ Tablas creadas exitosamente")
    
    print("📖 Cargando JSON...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"📊 Procesando {len(data)} comercializaciones...")
    db: Session = SessionLocal()

    try:
        for i, item in enumerate(data):
            if i % 100 == 0:  # Mostrar progreso cada 100 items
                print(f"   Procesando {i}/{len(data)}...")
                
            id_com = item["idComercializacion"]

            # Insertar Comercializacion
            com = models.Comercializacion(
                id=id_com,
                CodigoCotizacion=item.get("CodigoCotizacion"),
                ClienteId=item.get("ClienteId"),
                Cliente=item.get("NombreCliente"),
                LiderComercial=item.get("CorreoCreador"),
                FechaInicio=parse_fecha(item.get("FechaInicio")),
                ValorVenta=item.get("ValorFinalComercializacion"),
                ValorCotizacion=item.get("ValorFinalCotizacion"),
                NumeroEstados=item.get("NumeroEstados"),
                EsSENCE=1 if any(e["EstadoComercializacion"] == 3 for e in item.get("Estados", [])) else 0
            )
            db.merge(com)  # merge evita duplicados por ID

            # Insertar Estados
            for estado in item.get("Estados", []):
                est = models.Estado( 
                    idComercializacion=id_com,
                    EstadoComercializacion=estado.get("EstadoComercializacion"),
                    Fecha=parse_fecha(estado.get("Fecha"))
                )
                db.add(est)

            # Insertar Facturas
            for factura in item.get("Facturas", []):
                numero = factura.get("numero")
                for estado in factura.get("EstadosFactura", []):
                    fac = models.Factura(
                        idComercializacion=id_com,
                        CodigoCotizacion=item.get("CodigoCotizacion"),
                        NumeroFactura=numero,
                        FechaFacturacion=parse_fecha(factura.get("FechaFacturacion")),
                        NumeroEstadosFactura=factura.get("NumeroEstadosFactura"),
                        EstadoFactura=estado.get("estado"),
                        FechaEstado=parse_fecha(estado.get("Fecha")),
                        Pagado=estado.get("Pagado", 0),
                        Usuario=estado.get("Usuario"),
                        Observacion=estado.get("Observacion")
                    )
                    db.add(fac)
        
        print("💾 Guardando en base de datos...")
        db.commit()
        print("✅ Datos insertados en SQLite correctamente.")
        
    except Exception as e:
        print(f"❌ Error durante la carga: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 Iniciando carga de datos...")
    cargar_json_a_db("data/json_completo.json")
    print("🎉 Proceso completado!")