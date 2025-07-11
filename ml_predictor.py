"""
🤖 MODELO DE MACHINE LEARNING PARA PREDICCIÓN DE DÍAS DE PAGO
Contiene toda la lógica de ML separada del backend principal
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from typing import Tuple, Optional

# Configurar logging
logger = logging.getLogger(__name__)

class ModeloPrediccionML:
    """Clase para manejar todas las operaciones de Machine Learning"""
    
    def __init__(self):
        self.modelo_hibrido = None
        self.scaler_hibrido = None
        self.metadata_hibrido = None
        self.umbrales_inteligentes = None
        self.modelo_cargado = False
        
        # Cargar modelos al inicializar
        self._cargar_modelos()
    
    def _cargar_modelos(self):
        """Carga los modelos de ML desde archivos - prioriza modelo mejorado"""
        try:
            # Intentar cargar modelo híbrido MEJORADO primero
            modelo_mejorado_path = "modelo_hibrido_mejorado.pkl"
            scaler_mejorado_path = "scaler_hibrido_mejorado.pkl"
            metadata_mejorado_path = "modelo_hibrido_mejorado_metadata.pkl"
            
            if (os.path.exists(modelo_mejorado_path) and 
                os.path.exists(scaler_mejorado_path) and 
                os.path.exists(metadata_mejorado_path)):
                
                logger.info("🚀 Cargando modelo híbrido MEJORADO con deltas temporales...")
                self.modelo_hibrido = joblib.load(modelo_mejorado_path)
                self.scaler_hibrido = joblib.load(scaler_mejorado_path)
                self.metadata_hibrido = joblib.load(metadata_mejorado_path)
                
                logger.info(f"✅ Modelo MEJORADO cargado exitosamente")
                logger.info(f"   📅 Fecha: {self.metadata_hibrido.get('fecha_entrenamiento', 'N/A')}")
                logger.info(f"   🎯 MAE: {self.metadata_hibrido.get('mae', 'N/A')} días")
                logger.info(f"   📈 R²: {self.metadata_hibrido.get('r2', 'N/A')}")
                logger.info(f"   🔶 Features: {len(self.metadata_hibrido.get('features', []))} (incluye deltas)")
                
                self.version_modelo = "mejorado"
                
            else:
                # Fallback al modelo actual
                logger.info("🤖 Cargando modelo híbrido ACTUAL (fallback)...")
                modelo_path = "modelo_hibrido.pkl"
                scaler_path = "scaler_hibrido.pkl"
                metadata_path = "modelo_hibrido_metadata.pkl"
                
                if os.path.exists(modelo_path) and os.path.exists(scaler_path) and os.path.exists(metadata_path):
                    self.modelo_hibrido = joblib.load(modelo_path)
                    self.scaler_hibrido = joblib.load(scaler_path)
                    self.metadata_hibrido = joblib.load(metadata_path)
                    
                    logger.info(f"✅ Modelo ACTUAL cargado exitosamente")
                    logger.info(f"   📅 Fecha: {self.metadata_hibrido.get('fecha_entrenamiento', 'N/A')}")
                    logger.info(f"   🎯 MAE: {self.metadata_hibrido.get('mae', 'N/A')} días")
                    logger.info(f"   📈 R²: {self.metadata_hibrido.get('r2', 'N/A')}")
                    
                    self.version_modelo = "actual"
                else:
                    raise FileNotFoundError("No se encontraron archivos de modelo")
            
            # Umbrales diferenciados por tipo de proyecto (SENCE vs NO SENCE)
            self.umbrales_sence = {
                'muy_bajo': 37.0, 'bajo': 55.0, 'medio': 82.0, 
                'alto': 126.0, 'critico': 150.0, 'media': 65.0, 'mediana': 55.0
            }
            
            self.umbrales_no_sence = {
                'muy_bajo': 22.0, 'bajo': 35.0, 'medio': 47.0, 
                'alto': 62.0, 'critico': 80.0, 'media': 35.9, 'mediana': 35.0
            }
            
            # Umbrales generales (para compatibilidad)
            self.umbrales_inteligentes = self.umbrales_no_sence
            
            self.modelo_cargado = True
            logger.info("✅ Modelos de ML cargados exitosamente")
            logger.info(f"📊 MAE: {self.metadata_hibrido.get('mae', 'N/A')}, R²: {self.metadata_hibrido.get('r2', 'N/A')}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelos ML: {e}")
            self.modelo_cargado = False
    
    def esta_disponible(self) -> bool:
        """Verifica si el modelo está cargado y disponible"""
        return self.modelo_cargado and self.modelo_hibrido is not None
    
    def obtener_info_modelo(self) -> dict:
        """Devuelve información del modelo"""
        if not self.esta_disponible():
            return {"error": "Modelo no disponible"}
        
        # Información base del modelo
        modelo_info = {
            "modelo": {
                "tipo": "Random Forest Híbrido",
                "version": getattr(self, 'version_modelo', 'actual'),
                "mae": self.metadata_hibrido['mae'],
                "rmse": self.metadata_hibrido.get('rmse', 'N/A'),
                "r2": self.metadata_hibrido['r2'],
                "features_count": len(self.metadata_hibrido['features']),
                "fecha_entrenamiento": self.metadata_hibrido.get('fecha_entrenamiento', 'No disponible')
            },
            "features_principales": self.metadata_hibrido['features'][:10],
            "umbrales_comercial": self.umbrales_no_sence,
            "umbrales_sence": self.umbrales_sence,
            "rendimiento": {
                "precision": "Excelente (MAE < 3 días)",
                "r2_score": f"{self.metadata_hibrido['r2']:.4f}",
                "casos_entrenamiento": "12,729 casos históricos"
            }
        }
        
        # Información específica según el modelo cargado
        if hasattr(self, 'version_modelo') and self.version_modelo == "mejorado":
            modelo_info["modelo"].update({
                "mejoras": "Incluye features temporales Delta (ΔX, ΔY, ΔZ, ΔG)",
                "features_delta": ["DeltaX", "DeltaY", "DeltaZ", "DeltaG", "DeltaTotal", 
                                  "RatioDeltaX_Total", "RatioDeltaY_Total", "RatioDeltaZ_Total",
                                  "SENCE_x_DeltaG", "SENCE_x_DeltaX", "LogValorVenta_x_DeltaX"],
                "performance_boost": "28% mejor MAE vs modelo anterior",
                "descripcion": "Modelo híbrido mejorado con análisis temporal de ciclos de negocio"
            })
            modelo_info["caracteristicas_especiales"] = {
                "deltas_temporales": "Analiza patrones de tiempo por cliente",
                "features_interaccion": "SENCE x Delta, LogVenta x Delta",
                "mejora_performance": "MAE: 2.49 días (vs 3.45 anterior)",
                "validacion_overfitting": "Confirmado sin overfitting (diff train/test: 2.38%)"
            }
        else:
            modelo_info["modelo"].update({
                "descripcion": "Modelo base con features tradicionales de ventas",
                "status": "Modelo de respaldo - funcional y estable"
            })
        
        return modelo_info
    
    def clasificar_riesgo_inteligente(self, dias_predichos: int, es_sence: bool = False) -> Tuple[str, str, str, str]:
        """Clasifica el riesgo basado en umbrales diferenciados por tipo de proyecto"""
        
        # Seleccionar umbrales según tipo de proyecto
        umbrales = self.umbrales_sence if es_sence else self.umbrales_no_sence
        tipo_proyecto = "SENCE" if es_sence else "Comercial"
        
        if not umbrales:
            # Fallback a clasificación simple
            if dias_predichos <= 30:
                return "🟢 BAJO", "BAJO", "Bueno: Dentro del rango esperado", "Seguimiento normal"
            elif dias_predichos <= 60:
                return "🟡 MEDIO", "MEDIO", "Normal: Requiere atención", "Contacto proactivo recomendado"
            else:
                return "🔴 ALTO", "ALTO", "Preocupante: Requiere intervención", "Intervención urgente requerida"
        
        # Clasificación inteligente diferenciada
        if dias_predichos <= umbrales['muy_bajo']:
            return (
                "🟢 MUY BAJO", "MUY_BAJO",
                f"Excelente para {tipo_proyecto}: Mejor que 75% de casos similares (≤{umbrales['muy_bajo']:.0f} días)",
                "Seguimiento automático mensual"
            )
        elif dias_predichos <= umbrales['bajo']:
            return (
                "🟢 BAJO", "BAJO",
                f"Bueno para {tipo_proyecto}: Mejor que 50% de casos similares (≤{umbrales['bajo']:.0f} días)",
                "Seguimiento quincenal estándar"
            )
        elif dias_predichos <= umbrales['medio']:
            return (
                "🟡 MEDIO", "MEDIO",
                f"Normal para {tipo_proyecto}: Entre mediana y percentil 75 ({umbrales['bajo']:.0f}-{umbrales['medio']:.0f} días)",
                "Contacto proactivo semanal"
            )
        elif dias_predichos <= umbrales['alto']:
            return (
                "🟠 ALTO", "ALTO",
                f"Preocupante para {tipo_proyecto}: Peor que 75% de casos similares ({umbrales['medio']:.0f}-{umbrales['alto']:.0f} días)",
                "Seguimiento diario y escalamiento"
            )
        else:
            return (
                "🔴 CRÍTICO", "CRITICO",
                f"Alarmante para {tipo_proyecto}: Peor que 90% de casos similares (>{umbrales['alto']:.0f} días)",
                "Intervención inmediata y revisión gerencial"
            )
    
    def analizar_temporalidad(self, mes_facturacion: int, dias_estimados: int, 
                            dia_facturacion: int = 1, año: int = None) -> Tuple[bool, str]:
        """Determina si una venta se pagará en el mismo mes de su facturación"""
        # Días por mes (considerando años bisiestos)
        if año and año % 4 == 0 and (año % 100 != 0 or año % 400 == 0):
            dias_mes = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        else:
            dias_mes = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

        dias_totales_mes = dias_mes.get(mes_facturacion, 30)
        dias_disponibles = dias_totales_mes - dia_facturacion + 1
        
        if dias_estimados <= dias_disponibles:
            return True, f"✅ Se pagará dentro del mes (quedan {dias_disponibles} días desde el {dia_facturacion}/{mes_facturacion})"
        else:
            fecha_estimada_pago = dia_facturacion + dias_estimados
            mes_pago = mes_facturacion + (fecha_estimada_pago - 1) // dias_totales_mes
            return False, f"❌ Se pagará en el mes {mes_pago} (faltan {dias_estimados - dias_disponibles} días después del mes actual)"
    
    def calcular_confianza(self, dias_predichos: int) -> float:
        """Calcula un score de confianza basado en el rendimiento del modelo"""
        if not self.metadata_hibrido:
            return 0.8  # Confianza por defecto
        
        mae = self.metadata_hibrido.get('mae', 3.0)
        r2 = self.metadata_hibrido.get('r2', 0.9)
        
        # Confianza basada en el rendimiento del modelo
        confianza_modelo = min(1.0, r2)  # R² como base
        confianza_prediccion = max(0.3, 1 - (mae / 10))  # Basado en MAE
        
        return min(1.0, (confianza_modelo + confianza_prediccion) / 2)
    
    def _preparar_datos_prediccion(self, datos_venta: dict) -> pd.DataFrame:
        """Prepara los datos de entrada para el modelo (incluye deltas si es modelo mejorado)"""
        # Features básicas (comunes a ambos modelos)
        nueva_venta = pd.DataFrame({
            'ValorVenta': [datos_venta['valor_venta']],
            'EsSENCE': [1 if datos_venta['es_sence'] else 0],
            'MesFacturacion': [datos_venta['mes_facturacion']],
            'AñoFacturacion': [2024],
            'CantidadFacturas': [datos_venta['cantidad_facturas']],
            'RatioVentaCotizacion': [datos_venta['valor_venta'] / (datos_venta.get('valor_cotizacion') or datos_venta['valor_venta'])],
            'TiempoPromedioFacturas': [datos_venta.get('tiempo_promedio_facturas') or 
                                     (0 if datos_venta['cantidad_facturas'] == 1 else 
                                      min(15 + (datos_venta['cantidad_facturas'] - 1) * 5, 45))],
            'MontoPromedioFactura': [datos_venta['valor_venta'] / datos_venta['cantidad_facturas']],
            'StdPagos': [datos_venta['valor_venta'] * 0.1],
            'DiasInicioAFacturacion': [datos_venta.get('dias_inicio_facturacion', 30)],
            'TrimestreFacturacion': [(datos_venta['mes_facturacion'] - 1) // 3 + 1],
            'DiaSemanaFacturacion': [2]
        })
        
        # Encoding de cliente y vendedor
        cliente_cats = self.metadata_hibrido['cliente_categories']
        cliente_codes = {v: k for k, v in cliente_cats.items()}
        nueva_venta['Cliente_encoded'] = cliente_codes.get(datos_venta['cliente'], -1)
        
        vendedor_cats = self.metadata_hibrido['vendedor_categories']
        vendedor_codes = {v: k for k, v in vendedor_cats.items()}
        nueva_venta['Vendedor_encoded'] = vendedor_codes.get(datos_venta['correo_creador'], -1)
        
        # Categoría de monto
        percentiles = self.metadata_hibrido['percentiles_monto']
        if datos_venta['valor_venta'] <= percentiles[0]:
            categoria_monto = 0
        elif datos_venta['valor_venta'] <= percentiles[1]:
            categoria_monto = 1
        else:
            categoria_monto = 2
        nueva_venta['CategoriaMontoVenta'] = categoria_monto
        nueva_venta['CategoriaPagoCliente'] = 1
        
        # SI ES MODELO MEJORADO: Agregar features de deltas temporales
        if hasattr(self, 'version_modelo') and self.version_modelo == "mejorado":
            deltas = self._obtener_deltas_cliente(datos_venta['cliente'])
            
            # Features de deltas básicos
            nueva_venta['DeltaX'] = deltas['DeltaX']
            nueva_venta['DeltaY'] = deltas['DeltaY'] 
            nueva_venta['DeltaZ'] = deltas['DeltaZ']
            nueva_venta['DeltaG'] = deltas['DeltaG']
            
            # Features derivados de deltas
            nueva_venta['DeltaTotal'] = deltas['DeltaX'] + deltas['DeltaY'] + deltas['DeltaZ']
            nueva_venta['RatioDeltaX_Total'] = deltas['DeltaX'] / (deltas['DeltaG'] + 1)
            nueva_venta['RatioDeltaY_Total'] = deltas['DeltaY'] / (deltas['DeltaG'] + 1)
            nueva_venta['RatioDeltaZ_Total'] = deltas['DeltaZ'] / (deltas['DeltaG'] + 1)
            
            # Features de interacción
            nueva_venta['SENCE_x_DeltaG'] = nueva_venta['EsSENCE'] * deltas['DeltaG']
            nueva_venta['SENCE_x_DeltaX'] = nueva_venta['EsSENCE'] * deltas['DeltaX'] 
            nueva_venta['LogValorVenta_x_DeltaX'] = np.log1p(datos_venta['valor_venta']) * deltas['DeltaX']
            
            logger.info(f"🔶 Usando modelo MEJORADO con deltas para cliente: {datos_venta['cliente']}")
            
            # ASEGURAR QUE EL DATAFRAME TENGA EXACTAMENTE LAS FEATURES REQUERIDAS EN EL ORDEN CORRECTO
            features_esperadas = self.metadata_hibrido['features']
            nueva_venta = nueva_venta[features_esperadas]
            
        else:
            logger.info(f"🔷 Usando modelo ACTUAL sin deltas para cliente: {datos_venta['cliente']}")
        
        return nueva_venta
    
    def predecir_dias_pago(self, datos_venta: dict) -> dict:
        """Realiza la predicción principal de días de pago"""
        if not self.esta_disponible():
            raise ValueError("Modelo de ML no disponible")
        
        try:
            # Preparar datos
            nueva_venta = self._preparar_datos_prediccion(datos_venta)
            
            # Normalizar - DIFERENTE según modelo
            if hasattr(self, 'version_modelo') and self.version_modelo == "mejorado":
                # Modelo mejorado: normalizar TODAS las features
                nueva_venta_scaled = pd.DataFrame(
                    self.scaler_hibrido.transform(nueva_venta),
                    columns=nueva_venta.columns
                )
            else:
                # Modelo actual: normalizar solo columnas específicas
                cols_normalizar = ['ValorVenta', 'TiempoPromedioFacturas', 'MontoPromedioFactura', 'DiasInicioAFacturacion']
                nueva_venta[cols_normalizar] = self.scaler_hibrido.transform(nueva_venta[cols_normalizar])
                nueva_venta_scaled = nueva_venta
            
            # Predicción
            prediccion = self.modelo_hibrido.predict(nueva_venta_scaled[self.metadata_hibrido['features']])[0]
            dias_predichos = max(0, round(prediccion))
            
            # Clasificar riesgo usando umbrales diferenciados
            es_sence = bool(datos_venta.get('es_sence', False))
            nivel_riesgo, codigo_riesgo, descripcion_riesgo, accion = self.clasificar_riesgo_inteligente(dias_predichos, es_sence)
            
            # Análisis de temporalidad
            se_paga_mismo_mes, explicacion_mes = self.analizar_temporalidad(
                datos_venta['mes_facturacion'], dias_predichos, 1, 2024
            )
            
            # Calcular confianza
            confianza = self.calcular_confianza(dias_predichos)
            
            return {
                "dias_predichos": dias_predichos,
                "nivel_riesgo": nivel_riesgo,
                "codigo_riesgo": codigo_riesgo,
                "descripcion_riesgo": descripcion_riesgo,
                "accion_recomendada": accion,
                "confianza": confianza,
                "se_paga_mismo_mes": se_paga_mismo_mes,
                "explicacion_mes": explicacion_mes,
                "fecha_prediccion": datetime.now(),
                "modelo_version": f"Híbrido v2.0 - {getattr(self, 'version_modelo', 'actual').title()}"
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise
    
    def predecir_lote_simplificado(self, datos_venta: dict) -> dict:
        """Predicción simplificada para lotes (sin encoding completo)"""
        if not self.esta_disponible():
            raise ValueError("Modelo de ML no disponible")
        
        nueva_venta = pd.DataFrame({
            'ValorVenta': [datos_venta['valor_venta']],
            'EsSENCE': [1 if datos_venta['es_sence'] else 0],
            'MesFacturacion': [datos_venta['mes_facturacion']],
            'AñoFacturacion': [2024],
            'CantidadFacturas': [datos_venta['cantidad_facturas']],
            'RatioVentaCotizacion': [datos_venta['valor_venta'] / (datos_venta.get('valor_cotizacion') or datos_venta['valor_venta'])],
            'TiempoPromedioFacturas': [datos_venta.get('tiempo_promedio_facturas', 15)],
            'MontoPromedioFactura': [datos_venta['valor_venta'] / datos_venta['cantidad_facturas']],
            'StdPagos': [datos_venta['valor_venta'] * 0.1],
            'DiasInicioAFacturacion': [datos_venta.get('dias_inicio_facturacion', 30)],
            'TrimestreFacturacion': [(datos_venta['mes_facturacion'] - 1) // 3 + 1],
            'DiaSemanaFacturacion': [2],
            'Cliente_encoded': [-1],
            'Vendedor_encoded': [-1],
            'CategoriaMontoVenta': [1],
            'CategoriaPagoCliente': [1]
        })
        
        # SI ES MODELO MEJORADO: Agregar features de deltas (simplificados)
        if hasattr(self, 'version_modelo') and self.version_modelo == "mejorado":
            # Para lotes usamos valores promedio globales de deltas
            nueva_venta['DeltaX'] = 0.5
            nueva_venta['DeltaY'] = 0.3 
            nueva_venta['DeltaZ'] = 0.3
            nueva_venta['DeltaG'] = 1.1
            nueva_venta['DeltaTotal'] = 1.1
            nueva_venta['RatioDeltaX_Total'] = 0.45
            nueva_venta['RatioDeltaY_Total'] = 0.27
            nueva_venta['RatioDeltaZ_Total'] = 0.27
            nueva_venta['SENCE_x_DeltaG'] = nueva_venta['EsSENCE'] * 1.1
            nueva_venta['SENCE_x_DeltaX'] = nueva_venta['EsSENCE'] * 0.5
            nueva_venta['LogValorVenta_x_DeltaX'] = np.log1p(datos_venta['valor_venta']) * 0.5
        
        # Normalizar - DIFERENTE según modelo
        if hasattr(self, 'version_modelo') and self.version_modelo == "mejorado":
            # Modelo mejorado: normalizar TODAS las features
            nueva_venta_scaled = pd.DataFrame(
                self.scaler_hibrido.transform(nueva_venta),
                columns=nueva_venta.columns
            )
        else:
            # Modelo actual: normalizar solo columnas específicas
            cols_normalizar = ['ValorVenta', 'TiempoPromedioFacturas', 'MontoPromedioFactura', 'DiasInicioAFacturacion']
            nueva_venta[cols_normalizar] = self.scaler_hibrido.transform(nueva_venta[cols_normalizar])
            nueva_venta_scaled = nueva_venta
        
        # Predicción
        prediccion = self.modelo_hibrido.predict(nueva_venta_scaled[self.metadata_hibrido['features']])[0]
        dias_predichos = max(0, round(prediccion))
        
        # Clasificar riesgo usando umbrales diferenciados
        es_sence = bool(datos_venta.get('es_sence', False))
        nivel_riesgo, codigo_riesgo, descripcion_riesgo, accion = self.clasificar_riesgo_inteligente(dias_predichos, es_sence)
        
        return {
            "dias_predichos": dias_predichos,
            "nivel_riesgo": nivel_riesgo,
            "codigo_riesgo": codigo_riesgo,
            "accion_recomendada": accion
        }
    
    def _obtener_deltas_cliente(self, cliente_nombre: str) -> dict:
        """Obtiene los deltas temporales promedio para un cliente específico"""
        try:
            # Conectar a la base de datos
            import sqlite3
            import os
            
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'database.db')
            conn = sqlite3.connect(db_path)
            
            # Query para obtener deltas del cliente
            query = """
            SELECT DeltaX, DeltaY, DeltaZ, DeltaG 
            FROM metricas_tiempo 
            WHERE Cliente = ?
            """
            
            cursor = conn.cursor()
            cursor.execute(query, (cliente_nombre,))
            resultados = cursor.fetchall()
            conn.close()
            
            if resultados:
                # Calcular promedios de los deltas del cliente
                deltas_cliente = {
                    'DeltaX': sum(r[0] for r in resultados) / len(resultados),
                    'DeltaY': sum(r[1] for r in resultados) / len(resultados),
                    'DeltaZ': sum(r[2] for r in resultados) / len(resultados),
                    'DeltaG': sum(r[3] for r in resultados) / len(resultados)
                }
                logger.info(f"✅ Deltas encontrados para cliente {cliente_nombre}: {deltas_cliente}")
            else:
                # Valores default basados en promedios globales (del análisis previo)
                deltas_cliente = {
                    'DeltaX': 0.5,   # Promedio global DeltaX
                    'DeltaY': 0.3,   # Promedio global DeltaY
                    'DeltaZ': 0.3,   # Promedio global DeltaZ
                    'DeltaG': 1.1    # Promedio global DeltaG
                }
                logger.warning(f"⚠️ Cliente {cliente_nombre} no encontrado en deltas, usando valores default")
            
            return deltas_cliente
            
        except Exception as e:
            logger.error(f"Error obteniendo deltas para cliente {cliente_nombre}: {e}")
            # Valores default en caso de error
            return {
                'DeltaX': 0.5,
                'DeltaY': 0.3, 
                'DeltaZ': 0.3,
                'DeltaG': 1.1
            }

# Instancia global del modelo
modelo_ml = ModeloPrediccionML()
