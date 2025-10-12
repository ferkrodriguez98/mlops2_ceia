import os
import json
import time
import redis
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import csv
from datetime import datetime



import warnings
import logging

# === Suprimir warnings molestos ===
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# === Reducir el nivel de logs de MLflow y otros ===
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("pandas").setLevel(logging.ERROR)
logging.getLogger("botocore").setLevel(logging.ERROR)
logging.getLogger("boto3").setLevel(logging.ERROR)





# =====================
# Configuración general
# =====================
STREAM_KEY = "mlops_stream"
GROUP_NAME = "workers"
CONSUMER_NAME = "consumer-1"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

r = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

MODEL_NAME_MAP = {
    "knn": "Knn_Classifier",
    "svm": "SVC",
    "lightgbm": "LightGBM_Classifier",
}


# =====================
# Funciones reutilizadas
# =====================

def _resolve_registered_name(alias: str) -> str:
    name = MODEL_NAME_MAP.get(alias.lower())
    if not name:
        raise ValueError(f"Unknown model alias '{alias}'.")
    return name


def _get_latest_version_entry(reg_name: str):
    versions = client.get_latest_versions(reg_name)
    if not versions:
        all_versions = client.search_model_versions(f"name='{reg_name}'")
        if not all_versions:
            raise ValueError(f"No versions found for registered model '{reg_name}'.")
        versions = list(all_versions)
    best = max(versions, key=lambda v: int(v.version))
    return best


def _load_model_from_registry(alias: str):
    reg_name = _resolve_registered_name(alias)
    client.get_registered_model(reg_name)  # asegura que existe
    latest = _get_latest_version_entry(reg_name)
    model_uri = f"models:/{reg_name}/{latest.version}"
    print(f"[INFO] Loading model from URI: {model_uri}", flush=True)
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    return pyfunc_model


# =====================
# Inicialización
# =====================
print(f"✅ Conectado a Redis en {r}", flush=True)

try:
    r.xgroup_create(STREAM_KEY, GROUP_NAME, id="$", mkstream=True)
    print(f"✅ Grupo {GROUP_NAME} creado.")
except redis.exceptions.ResponseError:
    print(f"ℹ️ Grupo {GROUP_NAME} ya existía, continuando...")

print("🚀 Worker escuchando Redis Stream...")
print("⏳ Esperando nuevos mensajes...", flush=True)

# =====================
# Bucle principal
# =====================
while True:
    try:
        resp = r.xreadgroup(GROUP_NAME, CONSUMER_NAME, {STREAM_KEY: ">"}, count=1, block=5000)
        if not resp:
            continue

        for stream, messages in resp:
            for msg_id, msg_data in messages:
                payload = json.loads(msg_data["data"])
                model_name = payload["model"]
                print(f"📥 Mensaje recibido {msg_id} | Modelo: {model_name}", flush=True)

                try:
                    model = _load_model_from_registry(model_name)
                    df = pd.DataFrame(payload["payload"]["data"])
                    preds = model.predict(df)
                    print(f"✅ Predicción completada ({len(preds)} muestras): {preds.tolist()}", flush=True)
		    # === Guardar resultado en CSV ===
		    output_file = "/app/predicciones.csv"

		    # Asegurar que el archivo tenga encabezados si no existe
	   	    file_exists = os.path.isfile(output_file)

		    with open(output_file, mode="a", newline="", encoding="utf-8") as f:
		        writer = csv.writer(f)
 		       if not file_exists:
 		           writer.writerow(["timestamp", "modelo", "prediccion", "data"])

   		     writer.writerow([
		            datetime.now().isoformat(timespec="seconds"),
      		      reg_name,  # nombre del modelo (p.ej. LightGBM_Classifier)
     		       preds.tolist(),  # convierte array a lista legible
  		          df.to_json(orient="records")  # guarda los features
  		      ])

		    print(f"📝 Registro guardado en predicciones.csv para modelo {reg_name}")



                except Exception as e:
                    print(f"❌ Error procesando mensaje {msg_id}: {e}", flush=True)

                r.xack(STREAM_KEY, GROUP_NAME, msg_id)

    except Exception as e:
        print(f"⚠️ Error en loop principal: {e}", flush=True)
        time.sleep(5)



