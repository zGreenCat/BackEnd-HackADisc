import json
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models

def cargar_deltas_a_metricas(json_path="data/json_deltas_de_tiempo_por_cliente.json"):
    # Crear tabla si no existe
    print("🔧 Verificando tabla MetricasTiempo...")
    models.Base.metadata.create_all(bind=engine)
    
    print("📖 Cargando JSON con métricas...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metricas = data.get("metricas_generales", [])
    print(f"📊 Insertando {len(metricas)} registros...")

    db: Session = SessionLocal()
    try:
        for i, item in enumerate(metricas):
            if i % 50 == 0:
                print(f"   Procesando {i}/{len(metricas)}...")

            nueva_metrica = models.MetricasTiempo(
                idCliente=item.get("cliente_id"),
                Cliente=item.get("cliente").strip(),
                DeltaX=item.get("delta_x"),
                DeltaY=item.get("delta_y"),
                DeltaZ=item.get("delta_z"),
                DeltaG=item.get("delta_g"),
            )

            db.add(nueva_metrica)

        db.commit()
        print("✅ Datos insertados exitosamente en MetricasTiempo.")

    except Exception as e:
        print(f"❌ Error durante la inserción: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 Iniciando carga de métricas por cliente...")
    cargar_deltas_a_metricas()
    print("🎉 Proceso completado.")
