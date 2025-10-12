# Usa una imagen base de Python ligera (Recomendado para producción)
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requisitos e instala las dependencias
# Esto ayuda a aprovechar el caché de capas de Docker
COPY dockerfiles/grpc/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos de la aplicación
COPY . .

# Documenta el puerto en el que escucha el servicio gRPC (50051 es el estándar)
# NOTA: Esto solo documenta. Para acceso externo, debes usar -p 50051:50051 en el 'docker run'
EXPOSE 50051

# Define el comando para ejecutar el servidor gRPC de Python
# El script principal vive en grpc_service/server.py
CMD ["python", "grpc_service/server.py"]

# TESTING:
# docker build -t grpc:latest -f dockerfiles/grpc/Dockerfile .
# docker run -d -p 50051:50051 grpc