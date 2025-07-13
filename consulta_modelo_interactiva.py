#!/usr/bin/env python3
"""
Consulta interactiva del modelo ML
Permite probar predicciones con datos personalizados
"""

from ml_predictor import modelo_ml
import json
from datetime import datetime

def consulta_interactiva():
    """Interfaz interactiva para consultar el modelo"""
    print("CONSULTA DIRECTA AL MODELO ML")
    print("=" * 40)
    
    if not modelo_ml.esta_disponible():
        print("Modelo no disponible")
        return
    
    # Mostrar información del modelo
    info = modelo_ml.obtener_info_modelo()
    print(f"Modelo: {info.get('tipo', 'Unknown')}")
    print(f"MAE: {info.get('mae', 'N/A')}")
    print(f"R²: {info.get('r2', 'N/A')}")
    print("-" * 40)
    
    while True:
        print("\n¿Qué deseas hacer?")
        print("1. Predicción con datos personalizados")
        print("2. Predicción rápida con datos de ejemplo")
        print("3. Ver información del modelo")
        print("4. Salir")
        
        opcion = input("\nSelecciona una opción (1-4): ").strip()
        
        if opcion == "1":
            predecir_personalizado()
        elif opcion == "2":
            predecir_ejemplo()
        elif opcion == "3":
            mostrar_info_modelo()
        elif opcion == "4":
            print("¡Hasta luego!")
            break
        else:
            print("Opción inválida")

def predecir_personalizado():
    """Predicción con datos ingresados por el usuario"""
    try:
        print("\nIngresa los datos para la predicción:")
        
        cliente = input("Cliente: ").strip() or "CLIENTE_TEST"
        correo = input("Correo creador: ").strip() or "test@test.cl"
        
        valor_venta = input("Valor de venta: ").strip()
        valor_venta = int(valor_venta) if valor_venta.isdigit() else 500000
        
        es_sence = input("¿Es SENCE? (s/n): ").strip().lower() == 's'
        
        mes = input("Mes de facturación (1-12): ").strip()
        mes = int(mes) if mes.isdigit() and 1 <= int(mes) <= 12 else 7
        
        facturas = input("Cantidad de facturas: ").strip()
        facturas = int(facturas) if facturas.isdigit() else 1
        
        # Realizar predicción
        datos = {
            "cliente": cliente,
            "correo_creador": correo,
            "valor_venta": valor_venta,
            "es_sence": es_sence,
            "mes_facturacion": mes,
            "cantidad_facturas": facturas
        }
        
        resultado = modelo_ml.predecir_dias_pago(datos)
        mostrar_resultado(datos, resultado)
        
    except Exception as e:
        print(f"Error: {e}")

def predecir_ejemplo():
    """Predicción con datos de ejemplo predefinidos"""
    ejemplos = [
        {
            "nombre": "SENCE Grande",
            "datos": {
                "cliente": "EMPRESA_SENCE_GRANDE",
                "correo_creador": "ventas@empresa.cl",
                "valor_venta": 2000000,
                "es_sence": True,
                "mes_facturacion": 7,
                "cantidad_facturas": 3
            }
        },
        {
            "nombre": "Comercial Pequeño",
            "datos": {
                "cliente": "EMPRESA_COMERCIAL_PEQUEÑA",
                "correo_creador": "comercial@empresa.cl", 
                "valor_venta": 300000,
                "es_sence": False,
                "mes_facturacion": 3,
                "cantidad_facturas": 1
            }
        },
        {
            "nombre": "SENCE Mediano",
            "datos": {
                "cliente": "EMPRESA_SENCE_MEDIANA",
                "correo_creador": "admin@empresa.cl",
                "valor_venta": 800000,
                "es_sence": True,
                "mes_facturacion": 11,
                "cantidad_facturas": 2
            }
        }
    ]
    
    print("\nEjemplos disponibles:")
    for i, ejemplo in enumerate(ejemplos, 1):
        print(f"{i}. {ejemplo['nombre']}")
    
    try:
        seleccion = int(input("\nSelecciona un ejemplo (1-3): ")) - 1
        if 0 <= seleccion < len(ejemplos):
            ejemplo = ejemplos[seleccion]
            resultado = modelo_ml.predecir_dias_pago(ejemplo['datos'])
            print(f"\nProbando: {ejemplo['nombre']}")
            mostrar_resultado(ejemplo['datos'], resultado)
        else:
            print("Selección inválida")
    except ValueError:
        print("Ingresa un número válido")
    except Exception as e:
        print(f"Error: {e}")

def mostrar_resultado(datos_entrada, resultado):
    """Muestra el resultado de la predicción de forma clara"""
    print("\n" + "="*50)
    print("RESULTADO DE LA PREDICCIÓN")
    print("="*50)
    
    print("DATOS DE ENTRADA:")
    print(f"   Cliente: {datos_entrada['cliente']}")
    print(f"   Valor: ${datos_entrada['valor_venta']:,}")
    print(f"   SENCE: {'Sí' if datos_entrada['es_sence'] else 'No'}")
    print(f"   Mes: {datos_entrada['mes_facturacion']}")
    print(f"   Facturas: {datos_entrada['cantidad_facturas']}")
    
    print("\nRESULTADO:")
    print(f"   Días predichos: {resultado.get('dias_predichos', 'N/A')}")
    print(f"   Nivel de riesgo: {resultado.get('nivel_riesgo', 'N/A')}")
    print(f"   Código riesgo: {resultado.get('codigo_riesgo', 'N/A')}")
    print(f"   Confianza: {resultado.get('confianza', 'N/A')}")
    print(f"   Descripción: {resultado.get('descripcion_riesgo', 'N/A')}")
    print(f"   Acción recomendada: {resultado.get('accion_recomendada', 'N/A')}")
    print(f"   ¿Se paga mismo mes?: {'Sí' if resultado.get('se_paga_mismo_mes') else 'No'}")
    
    print("\nDETALLES TÉCNICOS:")
    print(f"   Modelo: {resultado.get('modelo_version', 'N/A')}")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def mostrar_info_modelo():
    """Muestra información detallada del modelo"""
    try:
        info = modelo_ml.obtener_info_modelo()
        print("\n" + "="*50)
        print("INFORMACIÓN DEL MODELO")
        print("="*50)
        
        for clave, valor in info.items():
            if isinstance(valor, dict):
                print(f"{clave}:")
                for subclave, subvalor in valor.items():
                    print(f"   {subclave}: {subvalor}")
            else:
                print(f"{clave}: {valor}")
    
    except Exception as e:
        print(f"Error obteniendo información: {e}")

if __name__ == "__main__":
    consulta_interactiva()
