"""
📊 ANÁLISIS PROFUNDO DEL MODELO ACTUAL
Genera gráficos y métricas para entender el comportamiento del modelo híbrido actual
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def cargar_modelo_actual():
    """Carga el modelo híbrido actual y sus metadatos"""
    print("🤖 Cargando modelo híbrido actual...")
    
    try:
        modelo = joblib.load('modelo_hibrido.pkl')
        scaler = joblib.load('scaler_hibrido.pkl')
        metadata = joblib.load('modelo_hibrido_metadata.pkl')
        
        print("✅ Modelo cargado exitosamente")
        print(f"   📅 Fecha entrenamiento: {metadata.get('fecha_entrenamiento', 'No disponible')}")
        print(f"   🎯 MAE: {metadata.get('mae', 'No disponible'):.2f} días")
        print(f"   📈 R²: {metadata.get('r2', 'No disponible'):.4f}")
        print(f"   🎲 OOB Score: {metadata.get('oob_score', 'No disponible'):.4f}")
        
        return modelo, scaler, metadata
    except FileNotFoundError as e:
        print(f"❌ Error: Archivo no encontrado - {e}")
        print("💡 Asegúrate de que el modelo esté entrenado y guardado")
        return None, None, None
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None, None, None

def analizar_feature_importance(modelo, metadata):
    """Analiza y visualiza la importancia de features"""
    print("\n🏆 ANÁLISIS DE IMPORTANCIA DE FEATURES")
    print("=" * 50)
    
    if 'feature_importance' not in metadata:
        print("❌ No hay datos de importancia de features en el modelo")
        return
    
    # Crear DataFrame de importancia
    importance_df = pd.DataFrame(metadata['feature_importance'])
    
    # Mostrar top 15 features
    print("📊 TOP 15 FEATURES MÁS IMPORTANTES:")
    for idx, row in importance_df.head(15).iterrows():
        print(f"   {idx+1:2d}. {row['feature']:25s}: {row['importance']:.4f}")
    
    # Gráfico de barras - Top 15 features
    plt.figure(figsize=(14, 8))
    top_15 = importance_df.head(15)
    
    bars = plt.barh(range(len(top_15)), top_15['importance'], 
                    color=['#e74c3c' if i == 0 else '#3498db' if i < 5 else '#95a5a6' 
                          for i in range(len(top_15))])
    
    plt.yticks(range(len(top_15)), top_15['feature'])
    plt.xlabel('Importancia')
    plt.title('🏆 Top 15 Features Más Importantes del Modelo Actual')
    plt.gca().invert_yaxis()
    
    # Añadir valores en las barras
    for i, v in enumerate(top_15['importance']):
        plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('feature_importance_actual.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico circular - Distribución de importancia por categorías
    plt.figure(figsize=(10, 8))
    
    # Categorizar features
    categorias = {
        'Temporales': ['MesFacturacion', 'AñoFacturacion', 'TrimestreFacturacion', 'DiaSemanaFacturacion', 
                      'TiempoPromedioFacturas', 'DiasInicioAFacturacion'],
        'Financieras': ['ValorVenta', 'MontoPromedioFactura', 'RatioVentaCotizacion', 'StdPagos'],
        'Cliente/Vendedor': ['Cliente_encoded', 'Vendedor_encoded', 'CategoriaPagoCliente'],
        'Características': ['EsSENCE', 'CantidadFacturas', 'CategoriaMontoVenta']
    }
    
    importancia_por_categoria = {}
    for categoria, features in categorias.items():
        total_importancia = importance_df[importance_df['feature'].isin(features)]['importance'].sum()
        importancia_por_categoria[categoria] = total_importancia
    
    plt.pie(importancia_por_categoria.values(), 
            labels=importancia_por_categoria.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6'])
    
    plt.title('📊 Distribución de Importancia por Categoría de Features')
    plt.axis('equal')
    plt.savefig('importancia_por_categoria.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df

def cargar_datos_para_analisis():
    """Carga los datos para análisis de correlaciones"""
    print("\n📊 Cargando datos para análisis de correlaciones...")
    
    try:
        ventas = pd.read_csv('ventas.csv')
        facturas = pd.read_csv('facturas.csv')
        
        # Convertir fechas
        ventas['FechaInicio'] = pd.to_datetime(ventas['FechaInicio'], errors='coerce')
        facturas['FechaFacturacion'] = pd.to_datetime(facturas['FechaFacturacion'], errors='coerce')
        facturas['FechaEstado'] = pd.to_datetime(facturas['FechaEstado'], errors='coerce')
        
        # Calcular días de pago
        pagos_stats = facturas.groupby('idComercializacion').agg({
            'Pagado': ['sum', 'count'],
            'FechaFacturacion': ['min', 'max'],
            'FechaEstado': ['min', 'max']
        }).reset_index()
        
        pagos_stats.columns = ['idComercializacion', 'TotalPagado', 'CantidadFacturas', 
                              'PrimeraFactura', 'UltimaFactura', 'PrimerPago', 'UltimoPago']
        
        pagos_stats['DiasPago'] = (pagos_stats['UltimoPago'] - pagos_stats['PrimeraFactura']).dt.days
        
        # Unir con ventas
        dataset = ventas.merge(pagos_stats, on='idComercializacion', how='inner')
        dataset = dataset[(dataset['DiasPago'] >= 0) & (dataset['DiasPago'] <= 300)].copy()
        
        print(f"✅ Dataset cargado: {len(dataset):,} registros")
        return dataset
    
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None

def analizar_correlaciones(dataset):
    """Analiza correlaciones entre variables y el target"""
    print("\n🔗 ANÁLISIS DE CORRELACIONES")
    print("=" * 30)
    
    # Seleccionar variables numéricas importantes
    vars_numericas = ['ValorVenta', 'ValorCotizacion', 'EsSENCE', 'NumeroEstados', 
                     'CantidadFacturas', 'TotalPagado', 'DiasPago']
    
    # Filtrar solo las que existen
    vars_existentes = [var for var in vars_numericas if var in dataset.columns]
    df_corr = dataset[vars_existentes].copy()
    
    # Calcular matriz de correlación
    correlation_matrix = df_corr.corr()
    
    # Gráfico de heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, fmt='.3f')
    plt.title('🔗 Matriz de Correlaciones - Variables Principales')
    plt.tight_layout()
    plt.savefig('correlaciones_matriz.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlaciones con DiasPago (nuestro target)
    if 'DiasPago' in correlation_matrix.columns:
        correlaciones_target = correlation_matrix['DiasPago'].abs().sort_values(ascending=False)
        
        print("🎯 CORRELACIONES CON DÍAS DE PAGO (TARGET):")
        for var, corr in correlaciones_target.items():
            if var != 'DiasPago':
                print(f"   {var:20s}: {corr:6.3f}")
        
        # Gráfico de correlaciones con target
        plt.figure(figsize=(10, 6))
        correlaciones_target_clean = correlaciones_target[correlaciones_target.index != 'DiasPago']
        
        colors = ['#e74c3c' if abs(x) > 0.5 else '#f39c12' if abs(x) > 0.3 else '#3498db' 
                 for x in correlaciones_target_clean.values]
        
        plt.barh(range(len(correlaciones_target_clean)), correlaciones_target_clean.values, color=colors)
        plt.yticks(range(len(correlaciones_target_clean)), correlaciones_target_clean.index)
        plt.xlabel('Correlación Absoluta con Días de Pago')
        plt.title('🎯 Correlaciones con el Target (Días de Pago)')
        plt.gca().invert_yaxis()
        
        # Líneas de referencia
        plt.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Correlación Moderada')
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Correlación Fuerte')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('correlaciones_target.png', dpi=300, bbox_inches='tight')
        plt.show()

def analizar_distribucion_target(dataset):
    """Analiza la distribución del target (DiasPago)"""
    print("\n📈 ANÁLISIS DE DISTRIBUCIÓN DEL TARGET")
    print("=" * 40)
    
    if 'DiasPago' not in dataset.columns:
        print("❌ No se encontró la columna DiasPago")
        return
    
    dias_pago = dataset['DiasPago'].dropna()
    
    print(f"📊 ESTADÍSTICAS DEL TARGET (DiasPago):")
    print(f"   Media: {dias_pago.mean():.2f} días")
    print(f"   Mediana: {dias_pago.median():.2f} días")
    print(f"   Desv. Estándar: {dias_pago.std():.2f} días")
    print(f"   Mínimo: {dias_pago.min():.0f} días")
    print(f"   Máximo: {dias_pago.max():.0f} días")
    print(f"   Q25: {dias_pago.quantile(0.25):.2f} días")
    print(f"   Q75: {dias_pago.quantile(0.75):.2f} días")
    
    # Gráfico de distribución
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histograma
    axes[0,0].hist(dias_pago, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(dias_pago.mean(), color='red', linestyle='--', label=f'Media: {dias_pago.mean():.1f}')
    axes[0,0].axvline(dias_pago.median(), color='orange', linestyle='--', label=f'Mediana: {dias_pago.median():.1f}')
    axes[0,0].set_title('📊 Distribución de Días de Pago')
    axes[0,0].set_xlabel('Días de Pago')
    axes[0,0].set_ylabel('Frecuencia')
    axes[0,0].legend()
    
    # Box plot
    axes[0,1].boxplot(dias_pago, vert=True)
    axes[0,1].set_title('📦 Box Plot - Días de Pago')
    axes[0,1].set_ylabel('Días de Pago')
    
    # Distribución por SENCE vs No SENCE
    if 'EsSENCE' in dataset.columns:
        sence_dias = dataset[dataset['EsSENCE'] == 1]['DiasPago'].dropna()
        no_sence_dias = dataset[dataset['EsSENCE'] == 0]['DiasPago'].dropna()
        
        axes[1,0].hist([no_sence_dias, sence_dias], bins=30, alpha=0.7, 
                      label=['No SENCE', 'SENCE'], color=['lightblue', 'lightcoral'])
        axes[1,0].set_title('🏢 Distribución por Tipo SENCE')
        axes[1,0].set_xlabel('Días de Pago')
        axes[1,0].set_ylabel('Frecuencia')
        axes[1,0].legend()
        
        print(f"\n🏢 COMPARACIÓN SENCE vs NO SENCE:")
        print(f"   No SENCE - Media: {no_sence_dias.mean():.2f} días")
        print(f"   SENCE - Media: {sence_dias.mean():.2f} días")
        print(f"   Diferencia: {sence_dias.mean() - no_sence_dias.mean():.2f} días")
    
    # Distribución por rangos de valor
    if 'ValorVenta' in dataset.columns:
        # Crear categorías de valor
        dataset['CategoriaValor'] = pd.cut(dataset['ValorVenta'], 
                                         bins=3, labels=['Bajo', 'Medio', 'Alto'])
        
        categorias = ['Bajo', 'Medio', 'Alto']
        datos_por_categoria = [dataset[dataset['CategoriaValor'] == cat]['DiasPago'].dropna() 
                              for cat in categorias]
        
        axes[1,1].boxplot(datos_por_categoria, labels=categorias)
        axes[1,1].set_title('💰 Días de Pago por Categoría de Valor')
        axes[1,1].set_xlabel('Categoría de Valor de Venta')
        axes[1,1].set_ylabel('Días de Pago')
    
    plt.tight_layout()
    plt.savefig('analisis_target_distribucion.png', dpi=300, bbox_inches='tight')
    plt.show()

def generar_reporte_completo(modelo, metadata, importance_df, dataset):
    """Genera un reporte completo del análisis"""
    print("\n📋 GENERANDO REPORTE COMPLETO")
    print("=" * 30)
    
    reporte = f"""
🤖 REPORTE DE ANÁLISIS DEL MODELO ACTUAL
{'='*50}

📅 INFORMACIÓN GENERAL:
   • Fecha de entrenamiento: {metadata.get('fecha_entrenamiento', 'No disponible')}
   • Casos de entrenamiento: {metadata.get('casos_entrenamiento', 'No disponible'):,}
   • Casos de test: {metadata.get('casos_test', 'No disponible'):,}

📊 MÉTRICAS DEL MODELO:
   • MAE (Error Absoluto Medio): {metadata.get('mae', 0):.2f} días
   • R² (Coeficiente de Determinación): {metadata.get('r2', 0):.4f}
   • RMSE (Raíz Error Cuadrático Medio): {metadata.get('rmse', 0):.2f} días
   • OOB Score: {metadata.get('oob_score', 0):.4f}

🏆 TOP 5 FEATURES MÁS IMPORTANTES:
"""
    
    for idx, row in importance_df.head(5).iterrows():
        reporte += f"   {idx+1}. {row['feature']}: {row['importance']:.4f}\n"
    
    if dataset is not None and 'DiasPago' in dataset.columns:
        dias_pago = dataset['DiasPago'].dropna()
        reporte += f"""
📈 ANÁLISIS DEL TARGET:
   • Media: {dias_pago.mean():.2f} días
   • Mediana: {dias_pago.median():.2f} días
   • Desviación Estándar: {dias_pago.std():.2f} días
   • Rango: {dias_pago.min():.0f} - {dias_pago.max():.0f} días
"""
    
    reporte += f"""
💡 RECOMENDACIONES:
   • El modelo tiene una precisión {'BUENA' if metadata.get('r2', 0) > 0.7 else 'MODERADA' if metadata.get('r2', 0) > 0.5 else 'BAJA'}
   • {'✅' if metadata.get('mae', 100) < 15 else '⚠️'} Error promedio: {metadata.get('mae', 0):.2f} días
   • Features temporales {'SÍ' if any('Tiempo' in feat or 'Mes' in feat or 'Año' in feat for feat in importance_df.head(5)['feature']) else 'NO'} están entre las más importantes
   • {'Considerar' if metadata.get('r2', 0) < 0.8 else 'No necesario'} re-entrenamiento con features temporales adicionales

📁 ARCHIVOS GENERADOS:
   • feature_importance_actual.png
   • importancia_por_categoria.png
   • correlaciones_matriz.png
   • correlaciones_target.png
   • analisis_target_distribucion.png
   • reporte_modelo_actual.txt
"""
    
    # Guardar reporte
    with open('reporte_modelo_actual.txt', 'w', encoding='utf-8') as f:
        f.write(reporte)
    
    print(reporte)
    print("✅ Reporte guardado en: reporte_modelo_actual.txt")

def main():
    """Función principal de análisis"""
    print("🔍 ANÁLISIS PROFUNDO DEL MODELO ACTUAL")
    print("=" * 50)
    
    # 1. Cargar modelo
    modelo, scaler, metadata = cargar_modelo_actual()
    if modelo is None:
        return
    
    # 2. Analizar importancia de features
    importance_df = analizar_feature_importance(modelo, metadata)
    if importance_df is None:
        return
    
    # 3. Cargar datos para análisis adicional
    dataset = cargar_datos_para_analisis()
    
    # 4. Análisis de correlaciones
    if dataset is not None:
        analizar_correlaciones(dataset)
        analizar_distribucion_target(dataset)
    
    # 5. Generar reporte completo
    generar_reporte_completo(modelo, metadata, importance_df, dataset)
    
    print("\n🎉 ANÁLISIS COMPLETADO")
    print("📊 Revisa los gráficos generados para tomar decisiones informadas")
    print("📋 Consulta el reporte completo en: reporte_modelo_actual.txt")

if __name__ == "__main__":
    main()
