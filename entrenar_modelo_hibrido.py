"""
🤖 MODELO HÍBRIDO AVANZADO - RE-ENTRENAMIENTO CON DATOS ACTUALIZADOS
Recreando el modelo híbrido original con los nuevos CSVs generados
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def cargar_datos():
    """Carga los CSVs actualizados"""
    print("📊 Cargando datos actualizados...")
    
    ventas = pd.read_csv('ventas.csv')
    facturas = pd.read_csv('facturas.csv')
    estados = pd.read_csv('estados.csv')
    
    print(f"✅ Ventas: {len(ventas):,} registros")
    print(f"✅ Facturas: {len(facturas):,} registros")
    print(f"✅ Estados: {len(estados):,} registros")
    
    return ventas, facturas, estados

def procesar_fechas(df, columnas_fecha):
    """Convierte columnas de fecha al formato datetime"""
    for col in columnas_fecha:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def crear_dataset_hibrido(ventas, facturas, estados):
    """Crea el dataset híbrido como el modelo original"""
    print("🔧 Creando dataset híbrido...")
    
    # Procesar fechas
    ventas = procesar_fechas(ventas, ['FechaInicio'])
    facturas = procesar_fechas(facturas, ['FechaFacturacion', 'FechaEstado'])
    estados = procesar_fechas(estados, ['Fecha'])
    
    # Calcular métricas de pago por comercialización
    pagos_stats = facturas.groupby('idComercializacion').agg({
        'Pagado': ['sum', 'mean', 'std', 'count'],
        'FechaFacturacion': ['min', 'max'],
        'FechaEstado': ['min', 'max']
    }).reset_index()
    
    # Aplanar nombres de columnas
    pagos_stats.columns = ['idComercializacion', 'TotalPagado', 'PromedioPagos', 'StdPagos', 
                          'CantidadFacturas', 'PrimeraFactura', 'UltimaFactura', 
                          'PrimerPago', 'UltimoPago']
    
    # Calcular días hasta pago completo
    pagos_stats['DiasPago'] = (pagos_stats['UltimoPago'] - pagos_stats['PrimeraFactura']).dt.days
    
    # Unir con ventas
    dataset = ventas.merge(pagos_stats, on='idComercializacion', how='inner')
    
    # Filtrar casos válidos
    dataset = dataset[
        (dataset['DiasPago'] >= 0) & 
        (dataset['DiasPago'] <= 300) &
        (dataset['ValorVenta'] > 0) &
        (dataset['TotalPagado'] > 0)
    ].copy()
    
    print(f"✅ Dataset híbrido: {len(dataset):,} casos válidos")
    return dataset

def crear_features_hibridas(dataset):
    """Crea las features híbridas avanzadas como el modelo original"""
    print("🎯 Creando features híbridas avanzadas...")
    
    # Features básicas
    dataset['ValorVenta'] = pd.to_numeric(dataset['ValorVenta'], errors='coerce')
    dataset['ValorCotizacion'] = pd.to_numeric(dataset['ValorCotizacion'], errors='coerce')
    dataset['EsSENCE'] = pd.to_numeric(dataset['EsSENCE'], errors='coerce')
    
    # Features de fecha
    dataset['MesFacturacion'] = dataset['FechaInicio'].dt.month
    dataset['AñoFacturacion'] = dataset['FechaInicio'].dt.year
    dataset['DiaSemanaFacturacion'] = dataset['FechaInicio'].dt.dayofweek
    dataset['TrimestreFacturacion'] = dataset['FechaInicio'].dt.quarter
    
    # Features derivadas financieras
    dataset['RatioVentaCotizacion'] = dataset['ValorVenta'] / dataset['ValorCotizacion'].replace(0, 1)
    dataset['MontoPromedioFactura'] = dataset['ValorVenta'] / dataset['CantidadFacturas']
    dataset['RatioPagadoVenta'] = dataset['TotalPagado'] / dataset['ValorVenta']
    
    # Features temporales
    dataset['TiempoPromedioFacturas'] = dataset['DiasPago'] / dataset['CantidadFacturas']
    dataset['DiasInicioAFacturacion'] = (dataset['PrimeraFactura'] - dataset['FechaInicio']).dt.days
    dataset['DiasInicioAFacturacion'] = dataset['DiasInicioAFacturacion'].fillna(30)
    
    # Encoding de clientes y vendedores con más casos
    clientes_freq = dataset['Cliente'].value_counts()
    vendedores_freq = dataset['LiderComercial'].value_counts()
    
    # Top clientes y vendedores (más conservador)
    top_clientes = clientes_freq.head(100).index
    top_vendedores = vendedores_freq.head(50).index
    
    cliente_map = {cliente: idx for idx, cliente in enumerate(top_clientes)}
    vendedor_map = {vendedor: idx for idx, vendedor in enumerate(top_vendedores)}
    
    dataset['Cliente_encoded'] = dataset['Cliente'].map(cliente_map).fillna(-1)
    dataset['Vendedor_encoded'] = dataset['LiderComercial'].map(vendedor_map).fillna(-1)
    
    # Categorías de monto (percentiles)
    percentiles_monto = dataset['ValorVenta'].quantile([0.33, 0.66]).values
    dataset['CategoriaMontoVenta'] = pd.cut(
        dataset['ValorVenta'], 
        bins=[0, percentiles_monto[0], percentiles_monto[1], float('inf')], 
        labels=[0, 1, 2]
    ).astype(int)
    
    # Categorías de comportamiento de pago por cliente
    cliente_stats = dataset.groupby('Cliente')['DiasPago'].mean()
    cliente_categorias = pd.cut(cliente_stats, bins=3, labels=[0, 1, 2])
    cliente_cat_map = cliente_categorias.to_dict()
    dataset['CategoriaPagoCliente'] = dataset['Cliente'].map(cliente_cat_map).fillna(1)
    
    # Features finales
    features = [
        'ValorVenta', 'EsSENCE', 'MesFacturacion', 'AñoFacturacion',
        'CantidadFacturas', 'RatioVentaCotizacion', 'TiempoPromedioFacturas',
        'MontoPromedioFactura', 'StdPagos', 'DiasInicioAFacturacion',
        'TrimestreFacturacion', 'DiaSemanaFacturacion',
        'Cliente_encoded', 'Vendedor_encoded', 'CategoriaMontoVenta', 'CategoriaPagoCliente'
    ]
    
    # Limpiar dataset
    dataset_clean = dataset[features + ['DiasPago']].dropna()
    
    print(f"✅ Features creadas: {len(features)} features, {len(dataset_clean):,} registros limpios")
    
    metadata = {
        'features': features,
        'cliente_categories': cliente_map,
        'vendedor_categories': vendedor_map,
        'percentiles_monto': percentiles_monto.tolist(),
        'total_clientes': len(top_clientes),
        'total_vendedores': len(top_vendedores)
    }
    
    return dataset_clean, features, metadata

def entrenar_modelo_hibrido(dataset, features, metadata):
    """Entrena el modelo híbrido Random Forest avanzado"""
    print("🚀 Entrenando modelo híbrido Random Forest...")
    
    X = dataset[features]
    y = dataset['DiasPago']
    
    print(f"📊 Datos de entrenamiento: {len(X):,} muestras, {len(features)} features")
    
    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5)
    )
    
    # Normalización solo para features numéricas sensibles
    cols_normalizar = ['ValorVenta', 'TiempoPromedioFacturas', 'MontoPromedioFactura', 'DiasInicioAFacturacion']
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_normalizar] = scaler.fit_transform(X_train[cols_normalizar])
    X_test_scaled[cols_normalizar] = scaler.transform(X_test[cols_normalizar])
    
    # Modelo Random Forest optimizado
    modelo = RandomForestRegressor(
        n_estimators=300,           # Más árboles
        max_depth=20,               # Profundidad mayor
        min_samples_split=5,        # Más restrictivo
        min_samples_leaf=2,         # Más restrictivo
        max_features='sqrt',        # Raíz cuadrada de features
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        warm_start=False
    )
    
    print("🏃‍♂️ Entrenando modelo híbrido...")
    modelo.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred_train = modelo.predict(X_train_scaled)
    y_pred_test = modelo.predict(X_test_scaled)
    
    # Métricas
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    oob_score = modelo.oob_score_
    
    print("📊 RESULTADOS DEL MODELO HÍBRIDO:")
    print(f"   🎯 MAE Training: {mae_train:.2f} días")
    print(f"   🎯 MAE Testing: {mae_test:.2f} días")
    print(f"   📈 R² Training: {r2_train:.4f}")
    print(f"   📈 R² Testing: {r2_test:.4f}")
    print(f"   📊 RMSE Testing: {rmse_test:.2f} días")
    print(f"   🎲 OOB Score: {oob_score:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': modelo.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🏆 TOP 10 FEATURES MÁS IMPORTANTES:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Metadata completa
    modelo_metadata = {
        'fecha_entrenamiento': datetime.now().isoformat(),
        'mae': mae_test,
        'r2': r2_test,
        'rmse': rmse_test,
        'oob_score': oob_score,
        'casos_entrenamiento': len(X_train),
        'casos_test': len(X_test),
        'feature_importance': feature_importance.to_dict('records'),
        **metadata
    }
    
    return modelo, scaler, modelo_metadata

def guardar_modelo_hibrido(modelo, scaler, metadata):
    """Guarda el modelo híbrido entrenado"""
    print("💾 Guardando modelo híbrido...")
    
    # Crear backup del modelo anterior si existe
    import os
    if os.path.exists('modelo_hibrido.pkl'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.rename('modelo_hibrido.pkl', f'modelo_hibrido_backup_{timestamp}.pkl')
        print(f"📦 Backup creado: modelo_hibrido_backup_{timestamp}.pkl")
    
    # Guardar nuevo modelo
    joblib.dump(modelo, 'modelo_hibrido.pkl')
    joblib.dump(scaler, 'scaler_hibrido.pkl')
    joblib.dump(metadata, 'modelo_hibrido_metadata.pkl')
    
    print("✅ Modelo híbrido guardado exitosamente")

def main():
    """Función principal de re-entrenamiento híbrido"""
    print("🚀 RE-ENTRENAMIENTO DEL MODELO HÍBRIDO CON DATOS ACTUALIZADOS")
    print("=" * 70)
    
    try:
        # 1. Cargar datos actualizados
        ventas, facturas, estados = cargar_datos()
        
        # 2. Crear dataset híbrido
        dataset = crear_dataset_hibrido(ventas, facturas, estados)
        
        if len(dataset) < 1000:
            print("❌ Error: Muy pocos datos para entrenar modelo híbrido")
            return
        
        # 3. Crear features híbridas
        dataset_clean, features, metadata = crear_features_hibridas(dataset)
        
        # 4. Entrenar modelo híbrido
        modelo, scaler, modelo_metadata = entrenar_modelo_hibrido(dataset_clean, features, metadata)
        
        # 5. Guardar modelo
        guardar_modelo_hibrido(modelo, scaler, modelo_metadata)
        
        print("=" * 70)
        print("🎉 RE-ENTRENAMIENTO HÍBRIDO COMPLETADO EXITOSAMENTE")
        print(f"📊 Precisión final: MAE={modelo_metadata['mae']:.2f} días, R²={modelo_metadata['r2']:.4f}")
        print(f"🎯 OOB Score: {modelo_metadata['oob_score']:.4f}")
        print("🚀 Modelo híbrido listo para predicciones de alta precisión")
        
    except Exception as e:
        print(f"❌ Error durante el re-entrenamiento híbrido: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
