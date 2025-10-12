# Streaming de datos en tiempo real con FastAPI, Redis y MLflow

## 1. Introducción
Este proyecto implementa un **sistema de inferencia en tiempo real** basado en una arquitectura distribuida compuesta por **FastAPI**, **Redis** y **MLflow**. Su objetivo es procesar flujos continuos de datos generados de manera sintética, enviarlos a través de un sistema de colas y obtener predicciones de modelos de machine learning registrados en MLflow.

El enfoque permite desacoplar la recepción de datos (API), el almacenamiento temporal (Redis Streams), el procesamiento asíncrono (Worker) y la gestión de modelos (MLflow), logrando un pipeline modular, escalable y fácilmente extensible.

---

## 2. Arquitectura general

```text
┌────────────────┐      ┌────────────────┐      ┌───────────────────────┐      ┌─────────────────┐
│  FastAPI       │ ---> │  Redis Stream  │ ---> │  Stream Worker (ML)   │ ---> │  MLflow Server  │
│ (API Gateway)  │      │  (Broker)      │      │  (Inferencia Asíncr.) │      │ (Model Registry)│
└────────────────┘      └────────────────┘      └───────────────────────┘      └─────────────────┘
       │                        │                          │                            │
       ▼                        ▼                          ▼                            ▼
   Usuario / Cliente     Mensajes JSON             Predicciones ML              Almacenamiento Modelo
```

---

## 3. Componentes del sistema

### 🔹 FastAPI (`fastapi`)
- Expone endpoints REST para recibir datos (`/predict_stream`).
- Publica los mensajes en **Redis Streams** bajo la clave `mlops_stream`.
- Actúa como punto de entrada al sistema.

### 🔹 Redis (`mlops2_ceia-redis-1`)
- Gestiona la cola de mensajes mediante **Streams**.
- Permite crear **grupos de consumidores** (e.g. `workers`), distribuyendo la carga de manera eficiente.

### 🔹 Stream Worker (`mlops2_ceia-stream_worker`)
- Consume los mensajes desde Redis.
- Carga los modelos desde MLflow (`SVC`, `Knn_Classifier`, `LightGBM_Classifier`).
- Realiza la inferencia y genera predicciones en tiempo real.
- Guarda los resultados en `predicciones.csv` y muestra logs de ejecución.

### 🔹 Fake Data Producer (`mlops2_ceia-fake_data_producer`)
- Genera datos sintéticos en base a estadísticas reales del dataset `enriched_employee_dataset.csv`.
- Envía registros simulados al endpoint `/predict_stream`.

### 🔹 MLflow (`mlflow`)
- Servidor de modelos (puerto 5001).
- Gestiona versiones y dependencias de modelos entrenados.

---

## 4. Flujo de datos paso a paso
1. **El generador** crea observaciones sintéticas y las envía a FastAPI vía HTTP POST.
2. **FastAPI** recibe la solicitud, la empaqueta en formato JSON y la publica en el stream `mlops_stream` de Redis.
3. **Redis** almacena el mensaje temporalmente y lo asigna a un grupo de consumidores (`workers`).
4. **Stream Worker** recupera el mensaje, identifica el modelo solicitado (e.g. `svm`, `knn`, `lightgbm`), y solicita el modelo correspondiente a MLflow.
5. **MLflow** devuelve el modelo registrado (última versión estable) y el worker realiza la inferencia.
6. **El resultado** se imprime en logs y se almacena en un archivo CSV con timestamp y predicción.

---

## 5. Ejecución con Docker Compose

### 🔧 Levantar todos los servicios
```bash
docker compose up --build
```

### 📊 Verificar contenedores activos
```bash
docker ps
```

### 🔍 Consultar logs del Worker
```bash
docker logs -f mlops2_ceia-stream_worker-1
```

### 🧠 Probar el flujo de datos manualmente
Enviar datos sintéticos a la API FastAPI:
```bash
curl -X POST "http://localhost:8800/predict_stream?model=svm" \
     -H "Content-Type: application/json" \
     -d '{"data": [{"Designation": 1.5, "Resource_Allocation": 1.7, "Sleep_Hours": 7}]}'
```

---

## 6. Logs reales de inferencia

**Worker** (conectado a Redis y MLflow):
```
📥 Mensaje recibido 1760296007925-0 | Modelo: knn
[INFO] Loading model from URI: models:/Knn_Classifier/1
❌ Error procesando mensaje: The feature names should match those that were passed during fit.

📥 Mensaje recibido 1760296014078-0 | Modelo: lightgbm
[INFO] Loading model from URI: models:/LightGBM_Classifier/1
✅ Predicción completada (1 muestras): [2]
```

---

## 7. Validación y monitoreo
- El **worker** imprime las predicciones directamente en consola.
- El archivo `predicciones.csv` almacena resultados persistentes para auditoría.
- Se puede inspeccionar el stream desde Redis CLI:
  ```bash
  docker exec -it mlops2_ceia-redis-1 redis-cli
  xlen mlops_stream
  ```

---

## 8. Conclusiones
- El sistema permite procesar flujos de datos en **tiempo real**, manteniendo el desacoplamiento entre los módulos de entrada, almacenamiento, procesamiento e inferencia.
- Su estructura facilita la **escalabilidad horizontal**, permitiendo múltiples workers concurrentes.
- Redis Streams ofrece un mecanismo robusto para **gestionar eventos** y **garantizar entrega**.
- MLflow centraliza la gestión de modelos y versiones, permitiendo replicar experimentos y mantener control sobre las dependencias.

Este enfoque constituye una base sólida para sistemas de **MLOps con inferencia en streaming**, aplicables tanto a datos sintéticos como a entornos de producción reales.

