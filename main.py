import os
import json
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging
import uvicorn
import threading
import time
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# ✅ Charger les variables d'environnement depuis Azure / .env
load_dotenv()

# ✅ Configuration d'Application Insights
instrumentation_key = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

if not instrumentation_key:
    raise ValueError("Azure Application Insights connection string is missing. Set APPLICATIONINSIGHTS_CONNECTION_STRING environment variable.")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    logger.addHandler(AzureLogHandler(connection_string=instrumentation_key))
except Exception as e:
    raise ValueError(f"Failed to set up AzureLogHandler: {str(e)}")

logger.info("Application Insights logging is configured successfully.")

# ✅ Configuration Azure Blob Storage
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
CONTAINER_NAME = "models"

if not AZURE_STORAGE_ACCOUNT_NAME or not AZURE_STORAGE_ACCOUNT_KEY:
    raise ValueError("Azure Storage account name or key is missing. Ensure AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY are set.")

blob_service_client = BlobServiceClient(
    f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
    credential=AZURE_STORAGE_ACCOUNT_KEY
)

# ✅ Répertoire temporaire pour stocker les modèles
tmp_model_dir = "/tmp/models"
os.makedirs(tmp_model_dir, exist_ok=True)

# ✅ Liste des fichiers à récupérer
model_files = [
    "best_model_fasttext.keras",
    "tv_layer_config.json",
    "tv_layer_vocabulary.txt"
]

def download_blob(file_name):
    """Télécharge un fichier blob depuis Azure Storage"""
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=file_name)
    local_file_path = os.path.join(tmp_model_dir, file_name)

    if not os.path.exists(local_file_path):
        logger.info(f"📥 Téléchargement de {file_name} depuis Azure Blob Storage...")
        with open(local_file_path, "wb") as file:
            file.write(blob_client.download_blob().readall())
        logger.info(f"✅ Fichier {file_name} téléchargé avec succès dans {local_file_path}")
    else:
        logger.info(f"✅ Fichier {file_name} déjà présent.")

# ✅ Télécharger tous les fichiers nécessaires
for file in model_files:
    download_blob(file)

# ✅ Définition des chemins
model_path = os.path.join(tmp_model_dir, "best_model_fasttext.keras")
config_path = os.path.join(tmp_model_dir, "tv_layer_config.json")
vocab_path = os.path.join(tmp_model_dir, "tv_layer_vocabulary.txt")

# ✅ Indicateur pour savoir si le modèle est chargé
model = None
tv_layer = None
model_loaded = False

def load_model():
    """Charge le modèle et la couche de vectorisation"""
    global model, tv_layer, model_loaded
    try:
        logger.info("🔄 Chargement du modèle en arrière-plan...")
        model = tf.keras.models.load_model(model_path)
        logger.info("✅ Modèle chargé avec succès.")

        # ✅ Chargement de la configuration TextVectorization
        with open(config_path, "r") as file:
            tv_layer_config = json.load(file)

        tv_layer = tf.keras.layers.TextVectorization.from_config(tv_layer_config)

        # ✅ Charger le vocabulaire
        with open(vocab_path, "r", encoding="utf-8") as vocab_file:
            vocabulary = [line.strip() for line in vocab_file]

        tv_layer.set_vocabulary(vocabulary)
        logger.info("✅ TextVectorization configuré avec succès.")
        model_loaded = True
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
        model_loaded = False

# ✅ Lancer le chargement du modèle en arrière-plan
threading.Thread(target=load_model, daemon=True).start()

# ✅ Initialiser FastAPI
app = FastAPI()

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url='/docs')

class TextInput(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    text: str
    prediction: str
    validation: bool

def wait_for_model():
    max_wait_time = 60
    waited = 0
    while not model_loaded and waited < max_wait_time:
        time.sleep(1)
        waited += 1
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading, please try again later.")

@app.post("/predict")
async def predict(input: TextInput):
    wait_for_model()
    try:
        sequences = tv_layer([input.text])
        prediction = model.predict(sequences)
        sentiment = "positive" if prediction[0][0] > 0.5 else "negative"

        logger.info(f"📢 Prédiction : {sentiment} pour le texte : {input.text}")
        return {"prediction": sentiment}
    except Exception as e:
        logger.error(f"❌ Erreur pendant la prédiction : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback(input: FeedbackInput):
    try:
        if not input.validation:
            logger.warning(f"⚠️ Tweet mal prédit : {input.text}, Prédiction : {input.prediction}")
        return {"message": "Feedback received, thank you!"}
    except Exception as e:
        logger.error(f"❌ Erreur lors du traitement du feedback : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Lancement du serveur FastAPI sur le port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
