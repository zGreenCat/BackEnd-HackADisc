# 🚀 HACKADISC - Backend Integrado con Predicciones ML

## 📋 Descripción del Proyecto

Sistema backend completo que integra gestión de datos empresariales con predicciones de Machine Learning para predecir días de pago de ventas.

## 🏗️ Estructura del Proyecto

```
HACKADISC/
└── BackEnd-HackADisc/            # Proyecto completo
    ├── README.md                 # Este archivo
    ├── main.py                   # 🚀 API FastAPI principal (endpoints)
    ├── ml_predictor.py           # 🤖 Lógica de Machine Learning
    ├── models.py                 # 📊 Modelos SQLAlchemy + Pydantic
    ├── database.py               # 🗄️ Configuración de base de datos
    ├── etl.py                    # 🔄 ETL para procesar datos
    ├── requirements.txt          # 📦 Dependencias Python
    ├── ventas.csv                # 📈 Datos originales de ventas
    ├── facturas.csv              # 🧾 Datos originales de facturas  
    ├── estados.csv               # 📋 Datos originales de estados
    ├── modelo_hibrido.pkl        # 🤖 Modelo ML entrenado
    ├── modelo_hibrido_metadata.pkl # 📊 Metadatos del modelo
    ├── scaler_hibrido.pkl        # ⚖️ Scaler para normalización
    ├── main_old.py               # 🗃️ Backup de versión anterior
    ├── .git/                     # 🔧 Control de versiones
    ├── .gitignore                # 🚫 Configuración git
    └── data/
        ├── database.db           # 🗃️ Base de datos SQLite
        └── json_completo.json    # 📄 Datos procesados
```

## 🎯 Funcionalidades

### 📊 **Endpoints de Datos Mejorados**
- `GET /` - Información general de la API
- `GET /health` - Health check del sistema
- `GET /resumen` - Resumen con totales de registros
- `GET /comercializaciones` - Lista de comercializaciones (con paginación)
- `GET /cliente/{nombre}` - Comercializaciones por cliente
- `GET /facturas` - Todas las facturas (con paginación)
- `GET /estados` - Todos los estados (con paginación)
- `GET /sence` - Estadísticas SENCE vs no-SENCE con porcentajes
- `GET /clientes` - Lista de todos los clientes únicos

### 🤖 **Endpoints de Machine Learning**
- `POST /predecir` - Predicción individual de días de pago
- `POST /predecir_lote` - Predicciones en lote (máx. 50)
- `GET /estadisticas_ml` - Estadísticas de predicciones
- `GET /modelo/info` - Información del modelo
- `GET /modelo/test` - Test rápido del modelo

## 🔧 Instalación y Ejecución

### 1. **Navegar al Directorio del Proyecto**
```bash
cd BackEnd-HackADisc
```

### 2. **Instalar Dependencias**
```bash
pip install -r requirements.txt
```

### 3. **Ejecutar ETL (si es necesario)**
```bash
python etl.py
```

### 4. **Iniciar el Servidor**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. **Acceder a la API**
- **API:** http://localhost:8000
- **Documentación:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## 📈 Modelo de Machine Learning

### **Características Técnicas**
- **Tipo:** Random Forest Híbrido
- **Precisión:** MAE: 2.38 días, R²: 0.915
- **Features:** 16 características para predicción
- **Casos de entrenamiento:** 12,729 casos históricos

### **Clasificación de Riesgo**
- 🟢 **MUY BAJO:** ≤28 días (mejor que 75% de casos)
- 🟢 **BAJO:** ≤36 días (mejor que 50% de casos)
- 🟡 **MEDIO:** 36-47 días (percentil 50-75)
- 🟠 **ALTO:** 47-63 días (percentil 75-90)
- 🔴 **CRÍTICO:** >63 días (peor que 90% de casos)

## 🔍 Ejemplo de Uso

### **Predicción Individual**
```bash
curl -X POST "http://localhost:8000/predecir" \
     -H "Content-Type: application/json" \
     -d '{
       "cliente": "INSECAP CAPACITACION PROFESIONAL",
       "correo_creador": "ventas@insecap.cl",
       "valor_venta": 1200000,
       "es_sence": true,
       "mes_facturacion": 7,
       "cantidad_facturas": 3
     }'
```

### **Respuesta Esperada**
```json
{
  "dias_predichos": 45,
  "nivel_riesgo": "🟡 MEDIO",
  "codigo_riesgo": "MEDIO",
  "descripcion_riesgo": "Normal: Entre mediana y percentil 75 (36-47 días)",
  "accion_recomendada": "Contacto proactivo semanal",
  "confianza": 0.84,
  "se_paga_mismo_mes": false,
  "explicacion_mes": "❌ Se pagará en el mes 8",
  "modelo_version": "Híbrido v2.0"
}
```

## 📦 Arquitectura del Sistema

### **🏗️ Separación de Responsabilidades**
- **`main.py`:** Endpoints y lógica de API
- **`ml_predictor.py`:** Toda la lógica de Machine Learning
- **`models.py`:** Modelos de datos (SQLAlchemy + Pydantic)
- **`database.py`:** Configuración de base de datos
- **`data/database.db`:** Fuente principal de datos (no CSV)

### **Tablas Principales**
- **comercializaciones:** Datos de ventas
- **facturas:** Información de facturación
- **estados:** Seguimiento de estados de venta

### **Estadísticas Actuales**
- **Comercializaciones:** 16,166 registros
- **Facturas:** 52,349 registros
- **Estados:** 33,837 registros

## 🛠️ Tecnologías Utilizadas

- **Backend:** FastAPI + SQLAlchemy
- **Base de Datos:** SQLite
- **ML:** scikit-learn (Random Forest)
- **Validación:** Pydantic
- **Logs:** Python logging
- **CORS:** Habilitado para frontends

## 📝 Notas Técnicas

- El modelo se carga automáticamente al iniciar la aplicación
- Cache en memoria para estadísticas de predicciones (últimas 1000)
- Validación robusta de datos de entrada
- Manejo de errores con mensajes descriptivos
- Background tasks para procesamiento asíncrono

## 🚀 Estado del Proyecto

✅ **COMPLETADO Y FUNCIONAL**
- Backend unificado operativo
- Modelo ML integrado y probado
- Todos los endpoints funcionando
- Documentación automática generada
- Base de datos con datos reales cargados

---

*Desarrollado para HACKADISC 2025*
