# MLOps2 - CEIA - Proyecto Final

## Integrantes del Grupo

- **Fermin Rodriguez del Castillo** (<ferkrodriguez98@gmail.com>)
- **Alejandro Lloveras** (<alejandro.lloveras@gmail.com>)
- **Ezequiel Maudet** (<eze.maudet@gmail.com>)
- **Gustavo Rivas** (<gus.j.rivas@gmail.com>)

## Servicios

- **Airflow:** http://localhost:8080
- **MLflow:** http://localhost:5001  
- **MinIO:** http://localhost:9001
- **API REST:** http://localhost:8800
- **API GraphQL:** http://localhost:8800/graphql
- **gRPC:** localhost:50051

# Dataset
https://www.kaggle.com/code/asanchezhernandez/employee-burnout-eda-and-prediction
Visión general del proyecto
*Repositorio de un pipeline MLOps orientado a predecir el “burn rate” de empleados, con servicios orquestados vía Docker Compose (Airflow, MLflow, MinIO, API REST/GraphQL, etc.).*

# Estructura de carpetas clave
airflow/: contiene los DAGs y plugins que definen el flujo de ingestión, limpieza, partición, ingeniería de variables y entrenamiento de modelos. Los DAGs orquestan la extracción desde MinIO, el registro de métricas en MLflow y la publicación de datasets procesados de vuelta en MinIO.

airflow/plugins/etl/: librería reutilizable con todas las transformaciones de datos (drop de columnas, imputación MICE, encoding categórico y target, escalado, etc.), compartidas entre los DAGs.

fastapi/: servicio de inferencia que expone REST y GraphQL; carga el último modelo registrado en MLflow Model Registry, entrega metadatos, predicciones y permite vaciar la caché local para recargar modelos.

dockerfiles/ y docker-compose.yaml: definen las imágenes personalizadas y la infraestructura local (Postgres, MinIO, MLflow, Airflow, Fastapi). Es fundamental para levantar todo el stack y reproducir el flujo de MLOps end-to-end.

data/: punto de montaje para el dataset enriquecido original que se sube a MinIO en el arranque del entorno.

Flujo de trabajo principal
Carga y preprocesamiento: el DAG etl_mlflow toma el dataset bruto desde MinIO, aplica todas las transformaciones, registra estadísticas en MLflow y publica los datasets entrenables (X/y train/test) en un bucket “processed”.

Entrenamiento: existen DAGs separados para cada modelo (LightGBM, KNN, SVM). El ejemplo de LightGBM lee los datos procesados, corre Optuna para buscar hiperparámetros, loguea métricas, guarda artefactos (como la matriz de confusión) y registra el modelo en MLflow Model Registry.

Serving: la API FastAPI obtiene dinámicamente el modelo más reciente del registro (mapeado por alias), ofrece endpoints de salud, listado de modelos, metadatos y predicciones, además de un endpoint GraphQL espejo de salud y modelos disponibles.

Conceptos importantes para quien se incorpora
Integración Airflow–MLflow–MinIO: entender cómo los DAGs interactúan con MinIO (extracción/carga de datos) y cómo se registran métricas, parámetros y artefactos en MLflow es esencial para extender o depurar el pipeline.

Reutilización de transformaciones: los helpers en airflow/plugins/etl/etl.py son el contrato para asegurar que inferencia y entrenamiento compartan el mismo pipeline de features (codificación, escalado, etc.). Cambios aquí impactan en todo el flujo.

Servicio de inferencia desacoplado: FastAPI actúa como consumer del Model Registry, por lo que cualquier modelo nuevo debe cumplir con el mismo esquema de features y registrarse correctamente para estar disponible vía API.

Infraestructura reproducible: docker-compose.yaml describe dependencias (Postgres para Airflow y MLflow, MinIO como backend S3). Saber levantar el stack completo es crítico para pruebas y demos.


