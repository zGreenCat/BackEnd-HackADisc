#!/usr/bin/env python3
"""
Script para validar el modelo ML directamente sin API
Permite probar predicciones y comparar con respuestas de la API
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json

def cargar_modelo():
    """Carga el modelo directamente desde los archivos"""
    try:
        # Cargar modelo, scaler y metadatos
        modelo = joblib.load('modelo_hibrido_mejorado.pkl')
        scaler = joblib.load('scaler_hibrido_mejorado.pkl')
        
        with open('modelo_hibrido_mejorado_metadata.pkl', 'rb') as f:
            metadata = joblib.load(f)
        
        print("Modelo cargado exitosamente")
        print(f"MAE: {metadata.get('mae', 'N/A')}")
        print(f"R²: {metadata.get('r2', 'N/A')}")
        print(f"Features: {len(metadata.get('features', []))}")
        
        return modelo, scaler, metadata
    
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None, None, None

def preparar_datos_prediccion(datos_entrada, metadata):
    """Prepara los datos de entrada para el modelo"""
    try:
        # Features esperadas por el modelo
        features_modelo = metadata.get('features', [])
        
        # Crear DataFrame con las características
        datos_df = pd.DataFrame([datos_entrada])
        
        # Asegurar que todas las features estén presentes
        for feature in features_modelo:
            if feature not in datos_df.columns:
                # Valores por defecto para features faltantes
                if 'mes' in feature.lower():
                    datos_df[feature] = datos_entrada.get('mes_facturacion', 7)
                elif 'sence' in feature.lower():
                    datos_df[feature] = int(datos_entrada.get('es_sence', False))
                elif 'valor' in feature.lower():
                    datos_df[feature] = datos_entrada.get('valor_venta', 500000)
                else:
                    datos_df[feature] = 0
        
        # Seleccionar solo las features del modelo en el orden correcto
        X = datos_df[features_modelo]
        
        print(f"Datos preparados: {X.shape}")
        print(f"Features utilizadas: {list(X.columns)}")
        
        return X
    
    except Exception as e:
        print(f"Error preparando datos: {e}")
        return None

def predecir_directo(modelo, scaler, X, metadata):
    """Realiza predicción directa con el modelo"""
    try:
        # Escalar datos
        X_scaled = scaler.transform(X)
        
        # Predicción
        prediccion = modelo.predict(X_scaled)[0]
        
        # Clasificar riesgo (usando misma lógica que ml_predictor.py)
        percentiles = metadata.get('percentiles', {
            'p25': 28, 'p50': 36, 'p75': 47, 'p90': 63
        })
        
        if prediccion <= percentiles['p25']:
            nivel_riesgo = "MUY BAJO"
        elif prediccion <= percentiles['p50']:
            nivel_riesgo = "BAJO"
        elif prediccion <= percentiles['p75']:
            nivel_riesgo = "MEDIO"
        elif prediccion <= percentiles['p90']:
            nivel_riesgo = "ALTO"
        else:
            nivel_riesgo = "CRÍTICO"
        
        return {
            "dias_predichos": round(prediccion),
            "nivel_riesgo": nivel_riesgo,
            "prediccion_raw": prediccion,
            "percentiles_usados": percentiles
        }
    
    except Exception as e:
        print(f"Error en predicción: {e}")
        return None

def comparar_con_api(datos_entrada, resultado_directo):
    """Compara resultado directo con respuesta de la API"""
    import requests
    
    try:
        # Preparar datos para la API
        datos_api = {
            "cliente": datos_entrada.get("cliente", "TEST_CLIENTE"),
            "correo_creador": datos_entrada.get("correo_creador", "test@test.cl"),
            "valor_venta": datos_entrada.get("valor_venta", 500000),
            "es_sence": datos_entrada.get("es_sence", False),
            "mes_facturacion": datos_entrada.get("mes_facturacion", 7),
            "cantidad_facturas": datos_entrada.get("cantidad_facturas", 1)
        }
        
        # Llamar a la API
        response = requests.post(
            "http://localhost:8000/predecir",
            json=datos_api,
            timeout=10
        )
        
        if response.status_code == 200:
            resultado_api = response.json()
            
            print("\nCOMPARACIÓN DE RESULTADOS:")
            print("=" * 50)
            print(f"Predicción Directa: {resultado_directo['dias_predichos']} días")
            print(f"Predicción API:     {resultado_api.get('dias_predichos', 'N/A')} días")
            print(f"Riesgo Directo:     {resultado_directo['nivel_riesgo']}")
            print(f"Riesgo API:         {resultado_api.get('nivel_riesgo', 'N/A')}")
            
            # Verificar coincidencia
            dias_coinciden = resultado_directo['dias_predichos'] == resultado_api.get('dias_predichos')
            riesgo_coincide = resultado_directo['nivel_riesgo'] == resultado_api.get('codigo_riesgo')
            
            print(f"\nDías coinciden: {dias_coinciden}")
            print(f"Riesgo coincide: {riesgo_coincide}")
            
            if not dias_coinciden or not riesgo_coincide:
                print(" DIFERENCIAS DETECTADAS - Revisar implementación")
            
            return resultado_api
        else:
            print(f"Error API: {response.status_code}")
            return None
    
    except Exception as e:
        print(f"Error comparando con API: {e}")
        return None

def main():
    """Función principal de validación"""
    print("VALIDADOR DIRECTO DEL MODELO ML")
    print("=" * 50)
    
    # 1. Cargar modelo
    modelo, scaler, metadata = cargar_modelo()
    if not modelo:
        return
    
    # 2. Datos de ejemplo para probar
    ejemplos_test = [
        {
            "cliente": "EMPRESA_TEST_1",
            "correo_creador": "test1@test.cl",
            "valor_venta": 1000000,
            "es_sence": True,
            "mes_facturacion": 7,
            "cantidad_facturas": 2
        },
        {
            "cliente": "EMPRESA_TEST_2", 
            "correo_creador": "test2@test.cl",
            "valor_venta": 500000,
            "es_sence": False,
            "mes_facturacion": 3,
            "cantidad_facturas": 1
        },
        {
            "cliente": "EMPRESA_TEST_3",
            "correo_creador": "test3@test.cl", 
            "valor_venta": 2000000,
            "es_sence": True,
            "mes_facturacion": 12,
            "cantidad_facturas": 4
        }
    ]
    
    # 3. Probar cada ejemplo
    for i, datos in enumerate(ejemplos_test, 1):
        print(f"\nPRUEBA {i}:")
        print("-" * 30)
        print(f"Cliente: {datos['cliente']}")
        print(f"Valor: ${datos['valor_venta']:,}")
        print(f"SENCE: {datos['es_sence']}")
        print(f"Mes: {datos['mes_facturacion']}")
        
        # Preparar datos
        X = preparar_datos_prediccion(datos, metadata)
        if X is None:
            continue
        
        # Predicción directa
        resultado = predecir_directo(modelo, scaler, X, metadata)
        if resultado:
            print(f"\nRESULTADO DIRECTO:")
            print(f"Días predichos: {resultado['dias_predichos']}")
            print(f"Nivel de riesgo: {resultado['nivel_riesgo']}")
            print(f"Valor raw: {resultado['prediccion_raw']:.2f}")
            
            # Comparar con API
            comparar_con_api(datos, resultado)
        
        print("-" * 50)
    
    print("\nValidación completada")

if __name__ == "__main__":
    main()
