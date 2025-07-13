# HACKADISC - Backend con ML Avanzado

## Descripción del Proyecto

Sistema backend completo desarrollado en FastAPI con predicciones de Machine Learning para estimar días de pago de comercializaciones. Incluye modelo híbrido Random Forest entrenado con datos históricos reales para predicción de ciclos de cobranza empresarial.

## Arquitectura del Sistema

### Estructura del Proyecto
```
BackEnd-HackADisc/
├── README.md                     # Documentación del proyecto
├── main.py                       # API FastAPI principal
├── ml_predictor.py               # Motor de Machine Learning
├── models.py                     # Modelos SQLAlchemy + Pydantic
├── database.py                   # Configuración de base de datos
├── requirements.txt              # Dependencias Python
├── generar_proyecciones_2025_2026.py  # Script de visualización
├── data/
│   └── database.db               # Base de datos SQLite principal
└── Modelos ML en Producción:
    ├── modelo_hibrido_mejorado.pkl         # Modelo Random Forest entrenado
    ├── modelo_hibrido_mejorado_metadata.pkl # Metadatos del modelo
    └── scaler_hibrido_mejorado.pkl         # Escalador de características
```

### Base de Datos
- **Comercializaciones:** 16,166 registros de ventas
- **Facturas:** 52,349 registros de facturación  
- **Estados:** 33,837 registros de seguimiento
- **Filtros aplicados:** Excluye códigos ADI%, OTR%, SPD% y solo incluye clientes que han pagado completamente

## Modelo de Machine Learning

### Características Técnicas
- **Algoritmo:** Random Forest Híbrido
- **Precisión:** MAE: 2.49 días, R²: 0.95
- **Casos de entrenamiento:** 12,729 comercializaciones históricas
- **Features utilizadas:** 16 características incluyendo valor de venta, tipo SENCE, mes de facturación, etc.

### Clasificación de Riesgo
- **MUY BAJO:** ≤28 días (75% de casos)
- **BAJO:** ≤36 días (50% de casos)  
- **MEDIO:** 36-47 días (percentil 50-75)
- **ALTO:** 47-63 días (percentil 75-90)
- **CRÍTICO:** >63 días (>90% de casos)

### Origen del Modelo
El modelo fue entrenado utilizando datos históricos reales de comercializaciones completamente pagadas, aplicando técnicas de ingeniería de características avanzadas y validación temporal para asegurar robustez predictiva en escenarios de producción.

**Propiedad Intelectual:** El modelo de Machine Learning, así como el algoritmo y metodología de entrenamiento utilizados, son propiedad intelectual exclusiva de Sebastian Concha M., ML Engineer del proyecto HACKADISC.

## API Endpoints

### Endpoints Principales
```http
GET /                              # Información general de la API
GET /health                        # Health check del sistema
```

### Gestión de Datos
```http
GET /resumen                       # Resumen con totales de registros
GET /comercializaciones            # Lista de comercializaciones (paginación)
GET /cliente/{nombre}              # Comercializaciones por cliente específico
GET /facturas                      # Todas las facturas (paginación)
GET /estados                       # Todos los estados (paginación)
GET /sence                         # Estadísticas SENCE vs no-SENCE
GET /clientes                      # Lista de todos los clientes únicos
GET /clientes/top                  # Clientes con estadísticas completas
```

### Machine Learning y Predicciones
```http
POST /predecir                     # Predicción individual de días de pago
POST /predecir_lote               # Predicciones en lote (máximo 50)
GET /estadisticas_ml              # Estadísticas de predicciones realizadas
GET /modelo/info                  # Información detallada del modelo
GET /modelo/test                  # Test rápido del modelo con datos de ejemplo
```

### Análisis y Proyecciones
```http
GET /prediccion_ingresos/{ano}/{mes}           # Predicción de ingresos mensuales
GET /prediccion_ingresos_resumen/{ano}/{mes}   # Resumen ejecutivo de ingresos
GET /cliente/{cliente_id}/probabilidad_pago_mes_actual  # Probabilidad de pago mensual
GET /proyeccion_anual/{ano}                    # Proyección anual completa
GET /proyeccion_trimestral/{ano}/{trimestre}   # Proyección trimestral
GET /proyeccion_anual_simplificada/{ano}       # Proyección anual simplificada
```

### Endpoints Especializados
```http
GET /cliente/{cliente_id}/vendedores           # Vendedores asociados a cliente
GET /clientes/estadisticas                     # Estadísticas detalladas de clientes
GET /clientes/con_vendedores                   # Clientes con información de vendedores
GET /clientes/con_confiabilidad               # Clientes con análisis de confiabilidad
```

## Instalación y Ejecución

### Requisitos Previos
- Python 3.8+
- pip

### Pasos de Instalación
```bash
# 1. Navegar al directorio del proyecto
cd BackEnd-HackADisc

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar el servidor
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Acceso a la API
- **API Base:** http://localhost:8000
- **Documentación Swagger:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## Ejemplo de Uso

### Predicción Individual
```bash
curl -X POST "http://localhost:8000/predecir" \
     -H "Content-Type: application/json" \
     -d '{
       "cliente": "EMPRESA EJEMPLO",
       "correo_creador": "ventas@empresa.cl",
       "valor_venta": 1200000,
       "es_sence": true,
       "mes_facturacion": 7,
       "cantidad_facturas": 2
     }'
```

### Respuesta de Predicción
```json
{
  "dias_predichos": 42,
  "nivel_riesgo": "MEDIO",
  "codigo_riesgo": "MEDIO",
  "descripcion_riesgo": "Normal: Entre mediana y percentil 75 (36-47 días)",
  "accion_recomendada": "Contacto proactivo semanal",
  "confianza": 0.87,
  "se_paga_mismo_mes": false,
  "explicacion_mes": "Se pagará en el mes 8",
  "modelo_version": "Híbrido v3.0"
}
```

### Proyección Anual Simplificada
```bash
curl "http://localhost:8000/proyeccion_anual_simplificada/2025"
```

## Tecnologías Utilizadas

### Backend
- **FastAPI:** Framework web moderno y rápido
- **SQLAlchemy:** ORM para manejo de base de datos
- **SQLite:** Base de datos ligera y eficiente
- **Pydantic:** Validación de datos y serialización

### Machine Learning
- **scikit-learn:** Algoritmos de ML (Random Forest)
- **pandas:** Manipulación de datos
- **numpy:** Computación numérica
- **joblib:** Serialización de modelos

### Visualización
- **matplotlib:** Generación de gráficos
- **requests:** Cliente HTTP para APIs

## Características Técnicas

### Seguridad y Robustez
- Validación estricta de datos de entrada con Pydantic
- Manejo de errores con mensajes descriptivos
- Filtrado automático de códigos excluidos (ADI%, OTR%, SPD%)
- Verificación de clientes completamente pagados

### Performance
- Cache en memoria para estadísticas de predicciones
- Background tasks para procesamiento asíncrono
- Paginación en endpoints de consulta masiva
- Límites configurables en consultas

### Funcionalidades Avanzadas
- Lógica condicional para datos históricos vs proyecciones
- Análisis temporal de ciclos de negocio
- Clasificación automática de riesgo de cobranza
- Proyecciones mensuales y anuales
- Análisis de confiabilidad por cliente

## Script de Visualización

El proyecto incluye `generar_proyecciones_2025_2026.py` que genera gráficos de proyecciones empresariales:

```bash
python generar_proyecciones_2025_2026.py
```

Genera visualizaciones de:
- Proyección mensual comparativa 2025-2026
- Totales anuales estimados
- Análisis de tendencias

## Validación y Consulta Directa del Modelo

### Scripts de Validación Incluidos

El proyecto incluye herramientas para validar y consultar directamente el modelo ML sin pasar por la API:

#### 1. Validación Completa del Modelo
```bash
python validar_modelo_directo.py
```

**Características:**
- Carga el modelo directamente desde archivos .pkl
- Ejecuta predicciones con datos de prueba predefinidos
- Compara resultados con respuestas de la API
- Detecta automáticamente discrepancias entre implementaciones
- Muestra métricas y percentiles utilizados

#### 2. Consulta Interactiva del Modelo
```bash
python consulta_modelo_interactiva.py
```

**Características:**
- Interfaz interactiva amigable para el usuario
- Permite ingresar datos personalizados
- Incluye ejemplos predefinidos para pruebas rápidas
- Muestra información detallada del modelo
- Ideal para validación manual y exploración

### Endpoints de Validación en la API

```bash
# Test rápido del modelo
curl "http://localhost:8000/modelo/test"

# Información detallada del modelo
curl "http://localhost:8000/modelo/info"

# Predicción específica para validar
curl -X POST "http://localhost:8000/predecir" \
     -H "Content-Type: application/json" \
     -d '{"cliente":"TEST","correo_creador":"test@test.cl","valor_venta":1000000,"es_sence":true,"mes_facturacion":7,"cantidad_facturas":2}'
```

### Casos de Uso para Validación

- **Validación técnica profunda:** Usar `validar_modelo_directo.py`
- **Pruebas interactivas:** Usar `consulta_modelo_interactiva.py`
- **Integración con sistemas:** Usar endpoints de la API
- **Debug de discrepancias:** Comparar resultados directos vs API


## Estado del Proyecto

**COMPLETADO Y FUNCIONAL**
- ✅ Backend unificado operativo
- ✅ Modelo ML integrado y validado (modelo_hibrido_mejorado.pkl)
- ✅ Todos los endpoints funcionando
- ✅ Documentación automática generada
- ✅ Base de datos con datos reales cargados
- ✅ Lógica de penalización implementada
- ✅ Proyecciones empresariales operativas

## Notas de Desarrollo

### Lógica de Penalización
- **Meses históricos:** Utilizan valor original de `valor_cobrado_real`
- **Meses de proyección:** Aplican penalización basada en la factura más reciente con estado 3

### Manejo de Datos
- Exclusión automática de comercializaciones con códigos ADI%, OTR%, SPD%
- Filtrado de clientes que no han completado pagos
- Fallbacks seguros para campos faltantes (CorreoCreador → LiderComercial)

### Nota Importante
- Cabe destacar que como es un prototipo, hay que probar mas los endpoint desarrollados, ya que en algunos casos pueden confundir al modelo y no dar la misma respuesta.

## Propiedad Intelectual

**IMPORTANTE:** El modelo de Machine Learning implementado en este proyecto, incluyendo:
- Algoritmo Random Forest híbrido y su configuración específica
- Metodología de entrenamiento y validación temporal
- Técnicas de ingeniería de características aplicadas
- Lógica de clasificación de riesgo y percentiles
- Archivos del modelo entrenado (.pkl)

**Son propiedad intelectual de Sebastian Concha M., ML Engineer.**

El uso, modificación, distribución o reproducción del modelo y sus algoritmos requiere autorización expresa del propietario intelectual.

---

**Desarrollado para HACKADISC 2025**

*Sebastian Concha M. / ML Engineer* 
*Fernando Condori / Full Stack Developer*
*Vicente Araya / Full Stack Developer*

