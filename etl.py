import pandas as pd
import json
from datetime import datetime
from sqlalchemy.orm import Session
from database import SessionLocal
import models

def parse_fecha(fecha_str):
    try:
        return datetime.strptime(fecha_str, "%d/%m/%Y")
    except:
        return None

def cargar_json_a_db(json_path="json_completo.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    db: Session = SessionLocal()

    for item in data:
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

    db.commit()
    db.close()
    print("Datos insertados en SQLite correctamente.")

if __name__ == "__main__":
    cargar_json_a_db("data/json_completo.json")