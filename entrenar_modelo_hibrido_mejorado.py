"""
🧪 MODELO HÍBRIDO MEJORADO CON DELTAS TEMPORALES
Usa exactamente la misma metodología del modelo actual pero agregando deltas como features adicionales
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import sqlite3
import warnings
warnings.filterwarnings('ignore')

def cargar_datos():
    """Carga datos desde la base de datos (como el modelo original hacía)"""
    print("📊 Cargando datos desde base de datos...")
    
    # Conectar a la base de datos
    conn = sqlite3.connect('data/database.db')
    
    # Cargar ventas (comercializaciones)
    ventas_query = """
    SELECT 
        id as idComercializacion,
        Cliente,
        LiderComercial,
        FechaInicio,
        ValorVenta,
        ValorCotizacion,
        EsSENCE,
        NumeroEstados
    FROM comercializaciones
    """
    ventas = pd.read_sql_query(ventas_query, conn)
    
    # Cargar facturas
    facturas_query = """
    SELECT 
        idComercializacion,
        FechaFacturacion,
        FechaEstado,
        Pagado
    FROM facturas
    """
    facturas = pd.read_sql_query(facturas_query, conn)
    
    # Cargar estados
    estados_query = """
    SELECT 
        idComercializacion,
        EstadoComercializacion as Estado,
        Fecha
    FROM estados
    """
    estados = pd.read_sql_query(estados_query, conn)
    
    # Cargar deltas temporales
    deltas_query = """
    SELECT 
        idCliente,
        Cliente as ClienteNombre,
        DeltaX,
        DeltaY,
        DeltaZ,
        DeltaG
    FROM metricas_tiempo
    WHERE idCliente IS NOT NULL
    """
    deltas = pd.read_sql_query(deltas_query, conn)
    conn.close()
    
    print(f"✅ Ventas: {len(ventas):,} registros")
    print(f"✅ Facturas: {len(facturas):,} registros")
    print(f"✅ Estados: {len(estados):,} registros")
    print(f"✅ Deltas temporales: {len(deltas):,} registros")
    
    return ventas, facturas, estados, deltas

def procesar_fechas(df, columnas_fecha):
    """Convierte columnas de fecha al formato datetime"""
    for col in columnas_fecha:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def crear_dataset_hibrido_mejorado(ventas, facturas, estados, deltas):
    """Crea el dataset híbrido IGUAL que el modelo actual pero con deltas"""
    print("🔧 Creando dataset híbrido mejorado...")
    
    # Procesar fechas EXACTAMENTE como el modelo original
    ventas = procesar_fechas(ventas, ['FechaInicio'])
    facturas = procesar_fechas(facturas, ['FechaFacturacion', 'FechaEstado'])
    estados = procesar_fechas(estados, ['Fecha'])
    
    # Calcular métricas de pago EXACTAMENTE como el modelo original
    pagos_stats = facturas.groupby('idComercializacion').agg({
        'Pagado': ['sum', 'mean', 'std', 'count'],
        'FechaFacturacion': ['min', 'max'],
        'FechaEstado': ['min', 'max']
    }).reset_index()
    
    # Aplanar nombres de columnas IGUAL que el original
    pagos_stats.columns = ['idComercializacion', 'TotalPagado', 'PromedioPagos', 'StdPagos', 
                          'CantidadFacturas', 'PrimeraFactura', 'UltimaFactura', 
                          'PrimerPago', 'UltimoPago']
    
    # Calcular días hasta pago completo IGUAL que el original
    pagos_stats['DiasPago'] = (pagos_stats['UltimoPago'] - pagos_stats['PrimeraFactura']).dt.days
    
    # Unir con ventas IGUAL que el original
    dataset = ventas.merge(pagos_stats, on='idComercializacion', how='inner')
    
    # Filtrar casos válidos EXACTAMENTE como el original
    dataset = dataset[
        (dataset['DiasPago'] >= 0) & 
        (dataset['DiasPago'] <= 300) &
        (dataset['ValorVenta'] > 0) &
        (dataset['TotalPagado'] > 0)
    ].copy()
    
    print(f"✅ Dataset híbrido base: {len(dataset):,} casos válidos")
    
    # AHORA AGREGAMOS LOS DELTAS TEMPORALES
    print("🔗 Uniendo con deltas temporales...")
    
    # Preparar mapeo de clientes para deltas
    deltas['ClienteNombre'] = deltas['ClienteNombre'].str.strip().str.upper()
    dataset['ClienteUpper'] = dataset['Cliente'].str.strip().str.upper()
    
    # Unir con deltas temporales
    dataset_con_deltas = dataset.merge(
        deltas[['ClienteNombre', 'DeltaX', 'DeltaY', 'DeltaZ', 'DeltaG']], 
        left_on='ClienteUpper', 
        right_on='ClienteNombre', 
        how='left'
    )
    
    print(f"📊 Dataset después de unir con deltas:")
    print(f"   Total registros: {len(dataset_con_deltas):,}")
    print(f"   Con deltas: {dataset_con_deltas['DeltaX'].notna().sum():,}")
    print(f"   Sin deltas: {dataset_con_deltas['DeltaX'].isna().sum():,}")
    
    # Rellenar valores faltantes de deltas con medianas globales
    for col in ['DeltaX', 'DeltaY', 'DeltaZ', 'DeltaG']:
        median_val = dataset_con_deltas[col].median()
        dataset_con_deltas[col] = dataset_con_deltas[col].fillna(median_val)
        print(f"   {col} - Mediana para filling: {median_val:.2f}")
    
    return dataset_con_deltas

def crear_features_hibridas_mejoradas(dataset):
    """Crea las features híbridas EXACTAMENTE como el original + deltas temporales"""
    print("🎯 Creando features híbridas mejoradas...")
    
    # Features básicas EXACTAMENTE como el original
    dataset['ValorVenta'] = pd.to_numeric(dataset['ValorVenta'], errors='coerce')
    dataset['ValorCotizacion'] = pd.to_numeric(dataset['ValorCotizacion'], errors='coerce')
    dataset['EsSENCE'] = pd.to_numeric(dataset['EsSENCE'], errors='coerce')
    
    # Features de fecha EXACTAMENTE como el original
    dataset['MesFacturacion'] = dataset['FechaInicio'].dt.month
    dataset['AñoFacturacion'] = dataset['FechaInicio'].dt.year
    dataset['DiaSemanaFacturacion'] = dataset['FechaInicio'].dt.dayofweek
    dataset['TrimestreFacturacion'] = dataset['FechaInicio'].dt.quarter
    
    # Features derivadas financieras EXACTAMENTE como el original
    dataset['RatioVentaCotizacion'] = dataset['ValorVenta'] / dataset['ValorCotizacion'].replace(0, 1)
    dataset['MontoPromedioFactura'] = dataset['ValorVenta'] / dataset['CantidadFacturas']
    dataset['RatioPagadoVenta'] = dataset['TotalPagado'] / dataset['ValorVenta']
    
    # Features temporales EXACTAMENTE como el original
    dataset['TiempoPromedioFacturas'] = dataset['DiasPago'] / dataset['CantidadFacturas']
    dataset['DiasInicioAFacturacion'] = (dataset['PrimeraFactura'] - dataset['FechaInicio']).dt.days
    dataset['DiasInicioAFacturacion'] = dataset['DiasInicioAFacturacion'].fillna(30)
    
    # Encoding de clientes y vendedores EXACTAMENTE como el original
    clientes_freq = dataset['Cliente'].value_counts()
    vendedores_freq = dataset['LiderComercial'].value_counts()
    
    # Top clientes y vendedores IGUAL que el original
    top_clientes = clientes_freq.head(100).index
    top_vendedores = vendedores_freq.head(50).index
    
    cliente_map = {cliente: idx for idx, cliente in enumerate(top_clientes)}
    vendedor_map = {vendedor: idx for idx, vendedor in enumerate(top_vendedores)}
    
    dataset['Cliente_encoded'] = dataset['Cliente'].map(cliente_map).fillna(-1)
    dataset['Vendedor_encoded'] = dataset['LiderComercial'].map(vendedor_map).fillna(-1)
    
    # Categorías de monto EXACTAMENTE como el original
    percentiles_monto = dataset['ValorVenta'].quantile([0.33, 0.66]).values
    dataset['CategoriaMontoVenta'] = pd.cut(
        dataset['ValorVenta'], 
        bins=[0, percentiles_monto[0], percentiles_monto[1], float('inf')], 
        labels=[0, 1, 2]
    ).astype(int)
    
    # Categorías de comportamiento de pago EXACTAMENTE como el original
    cliente_stats = dataset.groupby('Cliente')['DiasPago'].mean()
    cliente_categorias = pd.cut(cliente_stats, bins=3, labels=[0, 1, 2])
    cliente_cat_map = cliente_categorias.to_dict()
    dataset['CategoriaPagoCliente'] = dataset['Cliente'].map(cliente_cat_map).fillna(1)
    
    # Features originales EXACTAMENTE como el modelo actual
    features_originales = [
        'ValorVenta', 'EsSENCE', 'MesFacturacion', 'AñoFacturacion',
        'CantidadFacturas', 'RatioVentaCotizacion', 'TiempoPromedioFacturas',
        'MontoPromedioFactura', 'StdPagos', 'DiasInicioAFacturacion',
        'TrimestreFacturacion', 'DiaSemanaFacturacion',
        'Cliente_encoded', 'Vendedor_encoded', 'CategoriaMontoVenta', 'CategoriaPagoCliente'
    ]
    
    # AHORA AGREGAMOS LAS FEATURES DE DELTAS TEMPORALES
    print("🚀 Agregando features de deltas temporales...")
    
    # Features de deltas básicos
    features_deltas = ['DeltaX', 'DeltaY', 'DeltaZ', 'DeltaG']
    
    # Features derivados de deltas (simples pero efectivos)
    dataset['DeltaTotal'] = dataset['DeltaX'] + dataset['DeltaY'] + dataset['DeltaZ']
    dataset['RatioDeltaX_Total'] = dataset['DeltaX'] / (dataset['DeltaG'] + 1)
    dataset['RatioDeltaY_Total'] = dataset['DeltaY'] / (dataset['DeltaG'] + 1)
    dataset['RatioDeltaZ_Total'] = dataset['DeltaZ'] / (dataset['DeltaG'] + 1)
    
    # Features de interacción con SENCE (muy importantes según el experimento anterior)
    dataset['SENCE_x_DeltaG'] = dataset['EsSENCE'] * dataset['DeltaG']
    dataset['SENCE_x_DeltaX'] = dataset['EsSENCE'] * dataset['DeltaX']
    
    # Features de interacción con valor de venta
    dataset['LogValorVenta_x_DeltaX'] = np.log1p(dataset['ValorVenta']) * dataset['DeltaX']
    
    features_deltas_derivados = [
        'DeltaTotal', 'RatioDeltaX_Total', 'RatioDeltaY_Total', 'RatioDeltaZ_Total',
        'SENCE_x_DeltaG', 'SENCE_x_DeltaX', 'LogValorVenta_x_DeltaX'
    ]
    
    # Features finales: originales + deltas básicos + deltas derivados
    features_finales = features_originales + features_deltas + features_deltas_derivados
    
    # Limpiar dataset IGUAL que el original
    dataset_clean = dataset[features_finales + ['DiasPago']].dropna()
    
    print(f"✅ Features mejoradas creadas:")
    print(f"   Features originales: {len(features_originales)}")
    print(f"   Features deltas básicos: {len(features_deltas)}")
    print(f"   Features deltas derivados: {len(features_deltas_derivados)}")
    print(f"   TOTAL features: {len(features_finales)}")
    print(f"   Registros limpios: {len(dataset_clean):,}")
    
    metadata = {
        'features': features_finales,
        'features_originales': features_originales,
        'features_deltas': features_deltas + features_deltas_derivados,
        'cliente_categories': cliente_map,
        'vendedor_categories': vendedor_map,
        'percentiles_monto': percentiles_monto.tolist(),
        'total_clientes': len(top_clientes),
        'total_vendedores': len(top_vendedores)
    }
    
    return dataset_clean, features_finales, metadata

def entrenar_modelo_hibrido_mejorado(dataset, features, metadata):
    """Entrena el modelo híbrido mejorado con EXACTAMENTE los mismos parámetros del original"""
    print("🤖 Entrenando modelo híbrido mejorado...")
    
    # Preparar datos
    X = dataset[features].copy()
    y = dataset['DiasPago'].copy()
    
    print(f"📊 Dataset de entrenamiento:")
    print(f"   Features: {len(features)}")
    print(f"   Registros: {len(X):,}")
    print(f"   Target promedio: {y.mean():.2f} días")
    print(f"   Target std: {y.std():.2f} días")
    
    # División train/test CON LA MISMA SEMILLA que el original
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Escalado IGUAL que el original
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # EXACTAMENTE los mismos parámetros del modelo original
    modelo_mejorado = RandomForestRegressor(
        n_estimators=300,  # Igual que el original
        max_depth=20,      # Igual que el original
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,   # Misma semilla
        oob_score=True,
        n_jobs=-1
    )
    
    print("🔄 Entrenando Random Forest mejorado...")
    modelo_mejorado.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred_train = modelo_mejorado.predict(X_train_scaled)
    y_pred_test = modelo_mejorado.predict(X_test_scaled)
    
    # Métricas
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    oob_score = modelo_mejorado.oob_score_
    
    print(f"\n📈 MÉTRICAS DEL MODELO HÍBRIDO MEJORADO:")
    print(f"   MAE Train: {mae_train:.2f} días")
    print(f"   MAE Test: {mae_test:.2f} días")
    print(f"   R² Train: {r2_train:.4f}")
    print(f"   R² Test: {r2_test:.4f}")
    print(f"   RMSE Test: {rmse_test:.2f} días")
    print(f"   OOB Score: {oob_score:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': modelo_mejorado.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 TOP 15 FEATURES MÁS IMPORTANTES:")
    for idx, row in feature_importance.head(15).iterrows():
        tipo = "🔶 DELTA" if any(delta in row['feature'] for delta in ['Delta', 'SENCE_x_']) else "🔷 ORIGINAL"
        print(f"   {idx+1:2d}. {row['feature']:30s}: {row['importance']:.4f} {tipo}")
    
    # Metadata mejorada
    metadata_mejorada = {
        **metadata,
        'fecha_entrenamiento': datetime.now().isoformat(),
        'casos_entrenamiento': len(X_train),
        'casos_test': len(X_test),
        'mae': mae_test,
        'r2': r2_test,
        'rmse': rmse_test,
        'oob_score': oob_score,
        'feature_importance': feature_importance.to_dict('records'),
        'tipo_modelo': 'hibrido_mejorado_con_deltas'
    }
    
    # Guardar modelo mejorado
    joblib.dump(modelo_mejorado, 'modelo_hibrido_mejorado.pkl')
    joblib.dump(scaler, 'scaler_hibrido_mejorado.pkl')
    joblib.dump(metadata_mejorada, 'modelo_hibrido_mejorado_metadata.pkl')
    
    print(f"\n💾 Modelo híbrido mejorado guardado como:")
    print(f"   • modelo_hibrido_mejorado.pkl")
    print(f"   • scaler_hibrido_mejorado.pkl") 
    print(f"   • modelo_hibrido_mejorado_metadata.pkl")
    
    return modelo_mejorado, scaler, metadata_mejorada, feature_importance

def comparar_modelos_final():
    """Compara el modelo actual vs el híbrido mejorado"""
    print("\n⚖️ COMPARACIÓN FINAL DE MODELOS")
    print("=" * 60)
    
    try:
        # Cargar modelo actual
        metadata_actual = joblib.load('modelo_hibrido_metadata.pkl')
        
        # Cargar modelo híbrido mejorado
        metadata_mejorado = joblib.load('modelo_hibrido_mejorado_metadata.pkl')
        
        print("📊 MODELO ACTUAL (HÍBRIDO ORIGINAL):")
        print(f"   MAE: {metadata_actual.get('mae', 0):.2f} días")
        print(f"   R²: {metadata_actual.get('r2', 0):.4f}")
        print(f"   OOB Score: {metadata_actual.get('oob_score', 0):.4f}")
        print(f"   Features: {len(metadata_actual.get('feature_importance', []))}")
        
        print("\n🚀 MODELO HÍBRIDO MEJORADO (CON DELTAS):")
        print(f"   MAE: {metadata_mejorado.get('mae', 0):.2f} días")
        print(f"   R²: {metadata_mejorado.get('r2', 0):.4f}")
        print(f"   OOB Score: {metadata_mejorado.get('oob_score', 0):.4f}")
        print(f"   Features: {len(metadata_mejorado.get('feature_importance', []))}")
        
        # Calcular mejoras
        mejora_mae = metadata_actual.get('mae', 0) - metadata_mejorado.get('mae', 0)
        mejora_r2 = metadata_mejorado.get('r2', 0) - metadata_actual.get('r2', 0)
        mejora_oob = metadata_mejorado.get('oob_score', 0) - metadata_actual.get('oob_score', 0)
        
        print(f"\n📈 MEJORAS:")
        print(f"   MAE: {mejora_mae:+.2f} días ({'✅ MEJOR' if mejora_mae > 0 else '❌ PEOR'})")
        print(f"   R²: {mejora_r2:+.4f} ({'✅ MEJOR' if mejora_r2 > 0 else '❌ PEOR'})")
        print(f"   OOB: {mejora_oob:+.4f} ({'✅ MEJOR' if mejora_oob > 0 else '❌ PEOR'})")
        
        # Análisis de features de deltas
        print(f"\n🔶 ANÁLISIS FEATURES DELTA EN MODELO MEJORADO:")
        mejorado_features = pd.DataFrame(metadata_mejorado['feature_importance'])
        deltas_features = mejorado_features[
            mejorado_features['feature'].str.contains('Delta|SENCE_x_|LogValor')
        ].head(8)
        
        for idx, row in deltas_features.iterrows():
            print(f"   {row['feature']:30s}: {row['importance']:.4f}")
        
        # Recomendación final
        print(f"\n💡 RECOMENDACIÓN FINAL:")
        if mejora_r2 > 0.01 and mejora_mae > 0.3:
            print(f"   🚀 IMPLEMENTAR MODELO HÍBRIDO MEJORADO")
            print(f"   ✅ Mejora significativa en precisión")
            print(f"   ✅ Los deltas temporales aportan valor real")
        elif mejora_r2 > 0.005 and mejora_mae > 0:
            print(f"   🤔 CONSIDERAR MODELO HÍBRIDO MEJORADO")
            print(f"   ⚠️ Mejora moderada, evaluar impacto en producción")
        else:
            print(f"   🔄 MANTENER MODELO ACTUAL")
            print(f"   ❌ Las mejoras no justifican el cambio")
            
    except FileNotFoundError as e:
        print(f"❌ Error comparando modelos: {e}")

def main():
    """Función principal"""
    print("🚀 MODELO HÍBRIDO MEJORADO CON DELTAS TEMPORALES")
    print("=" * 70)
    print("📋 Estrategia: Misma metodología del modelo actual + deltas temporales")
    print()
    
    # 1. Cargar datos (CSVs + deltas)
    ventas, facturas, estados, deltas = cargar_datos()
    
    # 2. Crear dataset híbrido mejorado
    dataset = crear_dataset_hibrido_mejorado(ventas, facturas, estados, deltas)
    
    # 3. Crear features híbridas mejoradas
    dataset_final, features, metadata = crear_features_hibridas_mejoradas(dataset)
    
    # 4. Entrenar modelo híbrido mejorado
    modelo, scaler, metadata_final, importance = entrenar_modelo_hibrido_mejorado(
        dataset_final, features, metadata
    )
    
    # 5. Comparar modelos
    comparar_modelos_final()
    
    print(f"\n🎉 EXPERIMENTO HÍBRIDO MEJORADO COMPLETADO")
    print(f"📊 Revisa las métricas para tomar la decisión final")

if __name__ == "__main__":
    main()
