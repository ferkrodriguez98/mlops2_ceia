# MLOps2 - CEIA - Proyecto Final
Repositorio de un pipeline MLOps orientado a predecir el “burn rate” de empleados, con servicios orquestados vía Docker Compose (Airflow, MLflow, MinIO, API REST/GraphQL, etc.).

## Integrantes del Grupo

- **Fermin Rodriguez del Castillo** (<ferkrodriguez98@gmail.com>)
- **Alejandro Lloveras** (<alejandro.lloveras@gmail.com>)
- **Ezequiel Maudet** (<eze.maudet@gmail.com>)
- **Gustavo Rivas** (<gus.j.rivas@gmail.com>)

# Dataset
https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out

# Flujo de trabajo principal
1. ***Carga y preprocesamiento:*** el DAG etl_mlflow toma el dataset bruto desde MinIO, aplica todas las transformaciones, registra estadísticas en MLflow y publica los datasets entrenables (X/y train/test) en un bucket “processed”.

2. ***Entrenamiento:*** existen DAGs separados para cada modelo (LightGBM, KNN, SVM). El ejemplo de LightGBM lee los datos procesados, corre Optuna para buscar hiperparámetros, loguea métricas, guarda artefactos (como la matriz de confusión) y registra el modelo en MLflow Model Registry.

3. ***Serving:*** la API FastAPI obtiene dinámicamente el modelo más reciente del registro (mapeado por alias), ofrece endpoints de salud, listado de modelos, metadatos y predicciones, además de un endpoint GraphQL espejo de salud y modelos disponibles.

    - Para más detalle sobre el funcionamiento del **Streaming** consultar la [documentación](fastapi/README.md) correspondiente.

## Servicios

- **Airflow:** http://localhost:8080
- **MLflow:** http://localhost:5001  
- **MinIO:** http://localhost:9001
- **API REST:** http://localhost:8800/docs
- **API GraphQL:** http://localhost:8800/graphql
- **gRPC:** http://localhost:50051

## Ejecución con Docker Compose
1. **Levantar todos los servicios:**

    > docker compose --profile all up -d --build

2. **Log In en la [interfaz de Airflow](http://localhost:8080):**
    - _`User: airflow`_
    - _`Password: airflow`_
3. **Ejecutar el DAG para el proceso de ETL (`etl_mlflow`)**
4. **Luego, correr los DAGs de entrenamiento:**
    - `train_knn`
    - `train_lightgbm`
    - `train_svm`
5. **Finalizado el entrenamiento, ver las predicciones de `stream_worker` (Streaming):**

    > docker logs -f mlops2_ceia-stream_worker-1

6. **Además, pueden hacerse inferencias manuales por la [API](http://localhost:8800/docs)**

7. **Para detener la ejecución:**

    > docker compose --profile all down -v

### Verificación
- **Ver los mensajes brutos en Redis:**

    > docker exec -it mlops2_ceia-redis-1 redis-cli

    > xlen mlops_stream

    > xrange mlops_stream - +

- **Ver archivo de predicciones:**

    > docker exec -it mlops2_ceia-stream_worker-1 bash

    > ls /app

    > cat /app/predicciones.csv

# Estructura del Repositorio

La estructura se organiza por responsabilidad del servicio:
```
.
├── airflow/                  # Contiene la configuración de Airflow (dags/, config/, logs/).
├── data/                     # Conjunto de datos base del proyecto.
│   └── enriched_employee_dataset.csv
├── dockerfiles/              # Contiene los Dockerfiles para la construcción de cada imagen de servicio.
│   ├── airflow/
│   ├── fastapi/
│   ├── grpc/
│   ├── mlflow/
│   └── postgres/
├── fastapi/                  # Lógica del servidor API REST (Serving - Producción/Despliegue).
│   ├── fake_data_generator.py # Genera los datos falsos
│   ├── fake_data_producer.py # Emula las consultas de usuario.
│   ├── main.py               # Punto de entrada principal de la API.
│   ├── predicciones.csv      # Almacenamiento del stream de datos
│   └── stream_worker.py      # Procesamiento asíncrono e ingesta de datos en streaming.
├── grpc_service/             # Lógica del servidor gRPC (Servicio de baja latencia).
├── utils/                    # Funciones de utilidad y scripts auxiliares.
├── .gitignore
├── docker-compose.yaml       # Definición y orquestación de los servicios (Airflow, DB, Serving APIs, etc.).
├── EDA.ipynb                 # Notebooks para el Análisis Exploratorio de Datos.
└── LICENSE
```

## Carpetas y Archivos clave
- **`airflow/`:** contiene los DAGs y plugins que definen el flujo de ingestión, limpieza, partición, ingeniería de variables y entrenamiento de modelos. Los DAGs orquestan la extracción desde MinIO, el registro de métricas en MLflow y la publicación de datasets procesados de vuelta en MinIO.

- **`airflow/plugins/etl/`:** librería reutilizable con todas las transformaciones de datos (drop de columnas, imputación MICE, encoding categórico y target, escalado, etc.), compartidas entre los DAGs.

- **`fastapi/`:** servicio de inferencia que expone REST y GraphQL; carga el último modelo registrado en MLflow Model Registry, entrega metadatos, predicciones y permite vaciar la caché local para recargar modelos.

- **`dockerfiles/` y `docker-compose.yaml`:** definen las imágenes personalizadas y la infraestructura local (Postgres, MinIO, MLflow, Airflow, Fastapi). Es fundamental para levantar todo el stack y reproducir el flujo de MLOps end-to-end.

- **`data/`:** punto de montaje para el dataset enriquecido original que se sube a MinIO en el arranque del entorno.

# Conceptos generales importantes
- **Integración Airflow–MLflow–MinIO:** entender cómo los DAGs interactúan con MinIO (extracción/carga de datos) y cómo se registran métricas, parámetros y artefactos en MLflow es esencial para extender o depurar el pipeline.

- **Reutilización de transformaciones:** los helpers en airflow/plugins/etl/etl.py son el contrato para asegurar que inferencia y entrenamiento compartan el mismo pipeline de features (codificación, escalado, etc.). Cambios aquí impactan en todo el flujo.

- **Servicio de inferencia desacoplado:** FastAPI actúa como consumer del Model Registry, por lo que cualquier modelo nuevo debe cumplir con el mismo esquema de features y registrarse correctamente para estar disponible vía API.

- **Infraestructura reproducible:** docker-compose.yaml describe dependencias (Postgres para Airflow y MLflow, MinIO como backend S3). Saber levantar el stack completo es crítico para pruebas y demos.