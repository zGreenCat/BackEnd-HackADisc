"""
💰 PREDICTOR DE INGRESOS UNIVERSAL
Predice cuánto dinero se recibirá en cualquier mes específico
"""

import sys
sys.path.append('.')

from database import SessionLocal
import models
from ml_predictor import modelo_ml
from datetime import datetime, timedelta
import calendar
from typing import List, Dict, Tuple, Optional

class PredictorIngresosMensuales:
    """Clase para predecir ingresos de cualquier mes"""
    
    def __init__(self):
        self.db = None
        
    def __enter__(self):
        self.db = SessionLocal()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.db:
            self.db.close()
    
    def obtener_comercializaciones_pendientes(self) -> List[Dict]:
        """Obtiene todas las comercializaciones pendientes de pago"""
        comercializaciones = self.db.query(models.Comercializacion).all()
        comercializaciones_pendientes = []
        
        for com in comercializaciones:
            facturas = self.db.query(models.Factura).filter(
                models.Factura.idComercializacion == com.id
            ).all()
            
            valor_pagado = sum(f.Pagado or 0 for f in facturas)
            valor_pendiente = com.ValorVenta - valor_pagado
            
            if valor_pendiente > 0:
                comercializaciones_pendientes.append({
                    'comercializacion': com,
                    'valor_total': com.ValorVenta,
                    'valor_pagado': valor_pagado,
                    'valor_pendiente': valor_pendiente,
                    'cantidad_facturas': len(facturas)
                })
        
        return comercializaciones_pendientes
    
    def predecir_ingresos_mes(self, año: int, mes: int, limite: int = 500) -> Dict:
        """
        Predice los ingresos para un mes específico
        
        Args:
            año: Año objetivo (ej: 2025)
            mes: Mes objetivo (1-12)
            limite: Máximo de comercializaciones a analizar
        
        Returns:
            Dict con las predicciones y análisis
        """
        
        # Validaciones
        if not (1 <= mes <= 12):
            raise ValueError("El mes debe estar entre 1 y 12")
        
        if not (2024 <= año <= 2030):
            raise ValueError("El año debe estar entre 2024 y 2030")
        
        if not modelo_ml.esta_disponible():
            raise ValueError("Modelo de ML no disponible")
        
        # Fechas del mes objetivo
        inicio_mes = datetime(año, mes, 1)
        ultimo_dia = calendar.monthrange(año, mes)[1]
        fin_mes = datetime(año, mes, ultimo_dia)
        
        # Obtener comercializaciones pendientes
        comercializaciones_pendientes = self.obtener_comercializaciones_pendientes()
        
        # Ordenar por valor pendiente (mayor impacto primero)
        comercializaciones_pendientes.sort(
            key=lambda x: x['valor_pendiente'], 
            reverse=True
        )
        
        # Limitar análisis
        comercializaciones_a_analizar = comercializaciones_pendientes[:limite]
        
        # Variables para el análisis
        predicciones_mes = []
        errores = 0
        
        print(f"🔮 Prediciendo ingresos para {calendar.month_name[mes]} {año}")
        print(f"📅 Período: {inicio_mes.strftime('%d/%m/%Y')} - {fin_mes.strftime('%d/%m/%Y')}")
        print(f"📊 Analizando {len(comercializaciones_a_analizar):,} comercializaciones...")
        
        for i, cp in enumerate(comercializaciones_a_analizar):
            try:
                com = cp['comercializacion']
                
                # Preparar datos para predicción
                # Asumir facturación en el mes anterior al objetivo
                mes_facturacion = mes - 1 if mes > 1 else 12
                año_facturacion = año if mes > 1 else año - 1
                
                datos_prediccion = {
                    'cliente': com.Cliente or f'CLIENTE_{com.id}',
                    'correo_creador': getattr(com, 'LiderComercial', None) or 'vendedor@insecap.cl',
                    'valor_venta': max(float(cp['valor_pendiente']), 1.0),  # Evitar valores 0
                    'es_sence': bool(com.EsSENCE),
                    'mes_facturacion': mes_facturacion,
                    'cantidad_facturas': max(cp['cantidad_facturas'], 1),
                    'dias_inicio_facturacion': 30
                }
                
                # Realizar predicción
                prediccion = modelo_ml.predecir_dias_pago(datos_prediccion)
                
                if prediccion:
                    dias_hasta_pago = prediccion['dias_predichos']
                    
                    # Calcular fecha estimada de pago
                    fecha_facturacion = datetime(año_facturacion, mes_facturacion, 15)  # Media del mes
                    fecha_pago_estimada = fecha_facturacion + timedelta(days=dias_hasta_pago)
                    
                    # Verificar si cae en el mes objetivo
                    if inicio_mes <= fecha_pago_estimada <= fin_mes:
                        predicciones_mes.append({
                            'comercializacion_id': com.id,
                            'cliente': com.Cliente,
                            'valor_total': cp['valor_total'],
                            'valor_pagado': cp['valor_pagado'],
                            'valor_pendiente': cp['valor_pendiente'],
                            'dias_predichos': dias_hasta_pago,
                            'fecha_pago_estimada': fecha_pago_estimada,
                            'fecha_pago_estimada_str': fecha_pago_estimada.strftime('%Y-%m-%d'),
                            'es_sence': bool(com.EsSENCE),
                            'riesgo': prediccion.get('codigo_riesgo', 'MEDIO'),
                            'confianza': prediccion.get('confianza', 0.8)
                        })
                
                # Mostrar progreso
                if (i + 1) % 100 == 0:
                    print(f"  Procesadas: {i+1:,}/{len(comercializaciones_a_analizar):,}")
                    
            except Exception as e:
                errores += 1
                continue
        
        # Calcular estadísticas
        valor_total_mes = sum(p['valor_pendiente'] for p in predicciones_mes)
        valor_total_pendiente = sum(cp['valor_pendiente'] for cp in comercializaciones_pendientes)
        
        # Análisis por SENCE
        predicciones_sence = [p for p in predicciones_mes if p['es_sence']]
        predicciones_no_sence = [p for p in predicciones_mes if not p['es_sence']]
        
        # Análisis por nivel de riesgo
        analisis_riesgo = {}
        for p in predicciones_mes:
            riesgo = p['riesgo']
            if riesgo not in analisis_riesgo:
                analisis_riesgo[riesgo] = {'cantidad': 0, 'valor': 0}
            analisis_riesgo[riesgo]['cantidad'] += 1
            analisis_riesgo[riesgo]['valor'] += p['valor_pendiente']
        
        # Top clientes
        clientes_valor = {}
        for p in predicciones_mes:
            cliente = p['cliente'] or 'SIN_NOMBRE'
            if cliente not in clientes_valor:
                clientes_valor[cliente] = 0
            clientes_valor[cliente] += p['valor_pendiente']
        
        top_clientes = sorted(clientes_valor.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Distribución por semanas
        distribucion_semanal = {1: [], 2: [], 3: [], 4: []}
        for p in predicciones_mes:
            dia = p['fecha_pago_estimada'].day
            semana = min(((dia - 1) // 7) + 1, 4)
            distribucion_semanal[semana].append(p)
        
        return {
            'periodo': {
                'año': año,
                'mes': mes,
                'nombre_mes': calendar.month_name[mes],
                'inicio': inicio_mes.isoformat(),
                'fin': fin_mes.isoformat()
            },
            'resumen_principal': {
                'valor_proyectado': valor_total_mes,
                'cantidad_pagos': len(predicciones_mes),
                'porcentaje_del_pendiente': round((valor_total_mes / valor_total_pendiente * 100), 2) if valor_total_pendiente > 0 else 0,
                'promedio_dias': round(sum(p['dias_predichos'] for p in predicciones_mes) / len(predicciones_mes), 1) if predicciones_mes else 0,
                'confianza_promedio': round(sum(p['confianza'] for p in predicciones_mes) / len(predicciones_mes), 3) if predicciones_mes else 0
            },
            'contexto': {
                'total_comercializaciones_pendientes': len(comercializaciones_pendientes),
                'valor_total_pendiente': valor_total_pendiente,
                'comercializaciones_analizadas': len(comercializaciones_a_analizar),
                'errores_prediccion': errores
            },
            'analisis_sence': {
                'sence': {
                    'cantidad': len(predicciones_sence),
                    'valor': sum(p['valor_pendiente'] for p in predicciones_sence)
                },
                'no_sence': {
                    'cantidad': len(predicciones_no_sence),
                    'valor': sum(p['valor_pendiente'] for p in predicciones_no_sence)
                }
            },
            'analisis_riesgo': analisis_riesgo,
            'top_clientes': [
                {'cliente': cliente, 'valor': valor}
                for cliente, valor in top_clientes
            ],
            'distribucion_semanal': {
                f'semana_{semana}': {
                    'cantidad': len(pagos),
                    'valor': sum(p['valor_pendiente'] for p in pagos)
                }
                for semana, pagos in distribucion_semanal.items() if pagos
            },
            'predicciones_detalle': predicciones_mes[:50],  # Primeras 50 para API
            'metadata': {
                'fecha_generacion': datetime.now().isoformat(),
                'modelo_ml_activo': True,
                'limite_analizado': limite
            }
        }
    
    def generar_reporte_consola(self, resultado: Dict):
        """Genera un reporte legible para consola"""
        periodo = resultado['periodo']
        resumen = resultado['resumen_principal']
        contexto = resultado['contexto']
        
        print(f"\n" + "="*70)
        print(f"💰 REPORTE DE INGRESOS PROYECTADOS")
        print(f"📅 {periodo['nombre_mes']} {periodo['año']}")
        print(f"="*70)
        
        print(f"\n🎯 PREDICCIÓN PRINCIPAL:")
        print(f"  💰 Dinero proyectado: ${resumen['valor_proyectado']:,.0f}")
        print(f"  📊 Cantidad de pagos: {resumen['cantidad_pagos']:,}")
        print(f"  📈 % del total pendiente: {resumen['porcentaje_del_pendiente']:.2f}%")
        print(f"  ⏱️  Promedio días pago: {resumen['promedio_dias']:.1f}")
        print(f"  🎯 Confianza modelo: {resumen['confianza_promedio']:.1%}")
        
        print(f"\n📊 CONTEXTO:")
        print(f"  Total pendientes: {contexto['total_comercializaciones_pendientes']:,}")
        print(f"  Valor pendiente: ${contexto['valor_total_pendiente']:,.0f}")
        print(f"  Analizadas: {contexto['comercializaciones_analizadas']:,}")
        
        sence = resultado['analisis_sence']
        print(f"\n🎓 ANÁLISIS SENCE:")
        print(f"  SENCE: {sence['sence']['cantidad']:,} pagos - ${sence['sence']['valor']:,.0f}")
        print(f"  No SENCE: {sence['no_sence']['cantidad']:,} pagos - ${sence['no_sence']['valor']:,.0f}")
        
        if resultado['top_clientes']:
            print(f"\n🏆 TOP 5 CLIENTES:")
            for i, cliente_info in enumerate(resultado['top_clientes'][:5], 1):
                print(f"  {i}. {cliente_info['cliente'][:40]:<40} ${cliente_info['valor']:>12,.0f}")
        
        if resultado['analisis_riesgo']:
            print(f"\n⚠️  NIVEL DE RIESGO:")
            for riesgo, datos in sorted(resultado['analisis_riesgo'].items()):
                print(f"  {riesgo}: {datos['cantidad']:,} pagos - ${datos['valor']:,.0f}")
        
        print(f"\n" + "="*70)


def main_script(año: int, mes: int, limite: int = 500):
    """Función principal para usar desde línea de comandos"""
    try:
        with PredictorIngresosMensuales() as predictor:
            resultado = predictor.predecir_ingresos_mes(año, mes, limite)
            predictor.generar_reporte_consola(resultado)
            
            print(f"\n🎯 CONCLUSIÓN:")
            print(f"💰 Se proyecta recibir ${resultado['resumen_principal']['valor_proyectado']:,.0f}")
            print(f"   en {resultado['periodo']['nombre_mes']} {resultado['periodo']['año']}")
            
            return resultado
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predictor de ingresos mensuales')
    parser.add_argument('año', type=int, help='Año a predecir (ej: 2025)')
    parser.add_argument('mes', type=int, help='Mes a predecir (1-12)')
    parser.add_argument('--limite', type=int, default=500, help='Límite de comercializaciones a analizar')
    
    args = parser.parse_args()
    
    main_script(args.año, args.mes, args.limite)
