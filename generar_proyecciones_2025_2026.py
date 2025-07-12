#!/usr/bin/env python3
"""
GENERADOR DE PROYECCIONES HACKADISC 2025-2026
Solución final para visualización de proyecciones empresariales
"""

import requests
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Configuración
BASE_URL = "http://localhost:8000"

def obtener_proyeccion(ano):
    """Obtiene datos de proyección del API"""
    try:
        response = requests.get(f"{BASE_URL}/proyeccion_anual_simplificada/{ano}", timeout=60)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"Error obteniendo datos {ano}: {e}")
        return None

def generar_grafico(datos_2025, datos_2026):
    """Genera gráfico de proyecciones"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('PROYECCIONES HACKADISC 2025-2026', fontsize=16, fontweight='bold')
    
    # Extraer valores mensuales
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
             'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    valores_2025 = []
    valores_2026 = []
    
    for mes in range(1, 13):
        mes_key = f"mes_{mes:02d}"
        
        # 2025
        if mes_key in datos_2025['valores_mensuales']:
            valor_2025 = datos_2025['valores_mensuales'][mes_key]['valor_estimado'] / 1_000_000
        else:
            valor_2025 = 0
        valores_2025.append(valor_2025)
        
        # 2026
        if mes_key in datos_2026['valores_mensuales']:
            valor_2026 = datos_2026['valores_mensuales'][mes_key]['valor_estimado'] / 1_000_000
        else:
            valor_2026 = 0
        valores_2026.append(valor_2026)
    
    # Gráfico 1: Proyección mensual
    ax1.plot(meses, valores_2025, 'o-', linewidth=3, label='2025', color='#2E8B57')
    ax1.plot(meses, valores_2026, 's-', linewidth=3, label='2026', color='#4169E1')
    ax1.set_title('Proyección Mensual (Millones CLP)', fontweight='bold')
    ax1.set_ylabel('Millones CLP')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Gráfico 2: Resumen anual
    anos = ['2025', '2026']
    totales = [
        datos_2025['resumen_anual']['valor_total_estimado'] / 1_000_000,
        datos_2026['resumen_anual']['valor_total_estimado'] / 1_000_000
    ]
    
    bars = ax2.bar(anos, totales, color=['#2E8B57', '#4169E1'], alpha=0.8)
    ax2.set_title('Total Anual Estimado', fontweight='bold')
    ax2.set_ylabel('Millones CLP')
    
    # Añadir valores en las barras
    for bar, total in zip(bars, totales):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                f'{total:,.0f}M', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar
    filename = f"proyecciones_hackadisc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado: {filename}")
    plt.show()

def main():
    """Función principal"""
    print("GENERANDO PROYECCIONES HACKADISC")
    print("=" * 40)
    
    # Obtener datos
    datos_2025 = obtener_proyeccion(2025)
    datos_2026 = obtener_proyeccion(2026)
    
    if not datos_2025 or not datos_2026:
        print("Error obteniendo datos")
        return
    
    # Mostrar resúmenes
    print(f"2025: ${datos_2025['resumen_anual']['valor_total_estimado']:,.0f}")
    print(f"2026: ${datos_2026['resumen_anual']['valor_total_estimado']:,.0f}")
    
    # Generar gráfico
    generar_grafico(datos_2025, datos_2026)
    print("Proceso completado")

if __name__ == "__main__":
    main()
