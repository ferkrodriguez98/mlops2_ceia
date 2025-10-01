# MLOps2 - CEIA - Proyecto Final

## 👥 Integrantes del Grupo

- **Fermin Rodriguez del Castillo** (<ferkrodriguez98@gmail.com>)

---

## 📄 Descripción del Proyecto

Este proyecto implementa un sistema de MLOps avanzado para la predicción de burnout en empleados corporativos, utilizando múltiples protocolos de comunicación y técnicas de machine learning federado.

### 🎯 Objetivo

Desarrollar un sistema productivo que permita:
- Predicción de burnout mediante múltiples protocolos (REST, GraphQL, gRPC, Streaming)
- Aprendizaje federado para mejorar modelos
- Seguridad robusta y escalabilidad

### 🏗️ Arquitectura

- **Orquestación:** Apache Airflow
- **MLOps:** MLflow
- **APIs:** FastAPI con múltiples protocolos
- **Almacenamiento:** MinIO (S3-compatible)
- **Base de datos:** PostgreSQL
- **Contenedores:** Docker

---

## 🚀 Instalación

```bash
# Clonar repositorio
git clone <repo-url>
cd mlops2_ceia

# Levantar servicios
docker compose --profile all up
```

## 📊 Servicios

- **Airflow:** http://localhost:8080
- **MLflow:** http://localhost:5001  
- **MinIO:** http://localhost:9001
- **API REST:** http://localhost:8800
- **API GraphQL:** http://localhost:8800/graphql
- **gRPC:** localhost:50051

---

## 🔧 Desarrollo

Este proyecto está en desarrollo activo. Los commits se realizan de forma incremental para mostrar el progreso del trabajo.

### Próximas funcionalidades:
- [ ] GraphQL endpoint
- [ ] gRPC service
- [ ] Streaming predictions
- [ ] Federated learning
- [ ] Security layers