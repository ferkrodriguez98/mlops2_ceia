import json
import time
import redis
import pandas as pd
from pathlib import Path
from fake_data_generator import prepare_feature_space, FakeDataGenerator, records_from_dataframe

# Variables de entorno o defaults
REDIS_HOST = "redis"
REDIS_PORT = 6379
STREAM_KEY = "mlops_stream"

# Cargar dataset base para simular datos reales
csv_path = Path("/app/data/enriched_employee_dataset.csv")
feature_space = prepare_feature_space(csv_path)

# Crear el generador
generator = FakeDataGenerator(feature_space, noise_scale=0.05, seed=42)

# Conectar a Redis
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
print(f"✅ Conectado a Redis en {REDIS_HOST}:{REDIS_PORT}")

# Enviar continuamente datos falsos
for batch in generator.stream(batch_size=1, limit=None):
    record = list(records_from_dataframe(batch))[0]

    # Publicar al stream en el mismo formato que espera FastAPI
    payload = {
        "payload": {"data": [record], "columns": list(record.keys())},
        "model": "lightgbm"  # podés cambiar a "svm", "knn", etc.
    }
    r.xadd(STREAM_KEY, {"data": json.dumps(payload)})

    print(f"📤 Dato simulado enviado a {STREAM_KEY}: {payload['model']}")
    time.sleep(2)  # pausa para simular streaming continuo
