import os
import json
import requests
import tensorflow as tf
import asyncio
import threading
from azure.storage.blob import BlobServiceClient
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Désactiver l'utilisation du GPU pour éviter les erreurs sur Azure
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Charger les variables d'environnement
load_dotenv()

# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api_logger")

# Définition des variables globales
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
CONTAINER_NAME = "models"

# Définir un chemin absolu compatible Windows et Linux
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Répertoire du script
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Vérifier si les identifiants Azure sont bien définis
if not AZURE_STORAGE_ACCOUNT_NAME or not AZURE_STORAGE_ACCOUNT_KEY:
    raise ValueError("❌ Les informations Azure Storage ne sont pas définies dans les variables d'environnement.")

def download_model_from_azure(blob_name):
    """Télécharge un modèle depuis Azure Blob Storage si non présent en local."""
    local_file_path = os.path.join(MODEL_DIR, blob_name)
    
    if os.path.exists(local_file_path):
        logger.info(f"✅ {blob_name} est déjà présent localement.")
        return local_file_path

    logger.info(f"⬇️ Téléchargement de {blob_name} depuis Azure...")

    try:
        blob_service_client = BlobServiceClient(
            account_url=f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
            credential=AZURE_STORAGE_ACCOUNT_KEY
        )
        blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, blob_name)

        with open(local_file_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        # Vérification après téléchargement
        if os.path.exists(local_file_path):
            file_size = os.path.getsize(local_file_path)
            logger.info(f"✅ {blob_name} téléchargé avec succès ! Taille : {file_size} octets")
        else:
            raise FileNotFoundError(f"🚨 Le fichier {blob_name} n'a pas été trouvé après téléchargement.")

    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement de {blob_name} : {e}")
        return None

    return local_file_path

def load_model():
    """Charge le modèle en arrière-plan après le démarrage de l'API."""
    global model
    model_path = os.path.abspath(os.path.join(MODEL_DIR, "best_model_fasttext.keras"))

    logger.info(f"📂 Tentative de chargement du modèle depuis : {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"🚨 ERREUR : Le fichier {model_path} est introuvable après téléchargement !")
        return

    try:
        model = tf.keras.models.load_model(model_path)
        logger.info("✅ Modèle chargé avec succès !")
        logger.info("📊 Structure du modèle :")
        model.summary(print_fn=lambda x: logger.info(x))
    except Exception as e:
        logger.error(f"❌ ERREUR lors du chargement du modèle : {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer

    logger.info("🚀 Démarrage de l'API...")

    # Liste des fichiers à récupérer
    model_files = ["best_model_fasttext.keras", "tokenizer_fasttext.json"]

    for model_file in model_files:
        download_model_from_azure(model_file)
        await asyncio.sleep(2)  # Attente pour éviter les conflits d'accès

    logger.info("✅ Tous les fichiers nécessaires sont prêts !")

    # === Chargement du modèle en arrière-plan ===
    threading.Thread(target=load_model, daemon=True).start()

    # === Chargement du tokenizer JSON ===
    tokenizer_path = os.path.abspath(os.path.join(MODEL_DIR, "tokenizer_fasttext.json"))
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"❌ Le fichier tokenizer {tokenizer_path} est introuvable.")

    try:
        logger.info(f"📂 Chargement du tokenizer depuis : {tokenizer_path}")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
            tokenizer = tokenizer_from_json(tokenizer_data)
        logger.info("✅ Tokenizer chargé avec succès !")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du tokenizer : {e}")
        raise ValueError(f"Erreur de chargement du tokenizer : {e}")

    yield  # Maintien du contexte FastAPI

    logger.info("🛑 Arrêt de l'API...")

# === Initialisation de FastAPI ===
app = FastAPI(lifespan=lifespan)

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url='/docs')

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: TextInput):
    try:
        if not input.text.strip():
            raise HTTPException(status_code=400, detail="❌ Le texte d'entrée est vide")

        logger.info(f"📝 Texte reçu : {input.text}")

        # Vérifier si le modèle est chargé avant de prédire
        if 'model' not in globals():
            raise HTTPException(status_code=503, detail="Le modèle est en cours de chargement, veuillez réessayer plus tard.")

        sequence = tokenizer.texts_to_sequences([input.text])
        sequence_padded = pad_sequences(sequence, maxlen=50, padding="post", truncating="post")

        logger.info(f"📊 Séquence transformée : {sequence_padded}")

        prediction = model.predict(sequence_padded)
        sentiment = "positive" if prediction[0][0] > 0.5 else "negative"

        logger.info(f"🔮 Prédiction : {sentiment}")

        return {"prediction": sentiment}

    except HTTPException as he:
        logger.warning(f"⚠️ Mauvaise requête : {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"❌ Erreur de prédiction : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur serveur lors de la prédiction")

# === Lancement de l'application FastAPI ===
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🌍 Lancement du serveur sur le port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
