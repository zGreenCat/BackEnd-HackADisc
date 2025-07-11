import json
from datetime import datetime
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models

def cargar_top50_a_db(json_path="data/json_top50.json"):
    print("🔧 Creando tabla MetricasTiempo si no existe...")
    models.Base.metadata.create_all(bind=engine)
    print("✅ Tabla verificada")

    print("📖 Cargando JSON Top 50...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    top_clientes = data.get("top_50_clientes", [])
    print(f"📊 Procesando {len(top_clientes)} clientes...")

    db: Session = SessionLocal()
    fecha_actual = datetime.today().replace(day=1)

    try:
        for i, cliente in enumerate(top_clientes):
            if i % 10 == 0:
                print(f"   Insertando cliente {i}/{len(top_clientes)}...")

            metrica = models.MetricasTiempo(
                idCliente=cliente.get("cliente_id"),
                Cliente=cliente.get("cliente"),
                DeltaX=None,
                DeltaY=None,
                DeltaZ=None,
                DeltaG=None,
                Date=fecha_actual
            )
            db.merge(metrica)

        db.commit()
        print("✅ Clientes insertados correctamente.")

    except Exception as e:
        db.rollback()
        print(f"❌ Error durante la inserción: {e}")
        raise

    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 Iniciando ETL Top 50 clientes...")
    cargar_top50_a_db("data/json_top50.json")
    print("🎉 ETL completado.")