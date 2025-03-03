import os
import json
import requests
import tensorflow as tf
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

# Charger les variables d'environnement depuis le fichier .env si présent
load_dotenv()

# === Configuration Azure Blob Storage ===
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
CONTAINER_NAME = "models"

if not AZURE_STORAGE_ACCOUNT_NAME or not AZURE_STORAGE_ACCOUNT_KEY:
    raise ValueError("Les informations Azure Storage ne sont pas définies dans les variables d'environnement.")

# Dossier local des modèles
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_model_from_azure(blob_name):
    """Télécharge un modèle depuis Azure Blob Storage si non présent en local."""
    local_file_path = os.path.join(MODEL_DIR, blob_name)
    if not os.path.exists(local_file_path):
        print(f"Téléchargement de {blob_name} depuis Azure...")
        try:
            blob_service_client = BlobServiceClient(
                account_url=f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
                credential=AZURE_STORAGE_ACCOUNT_KEY
            )
            blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, blob_name)
            with open(local_file_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            print(f"{blob_name} téléchargé avec succès !")
        except Exception as e:
            print(f"Erreur lors du téléchargement de {blob_name} : {e}")
            return None
    return local_file_path

# Liste des modèles à récupérer
model_files = [
    "best_model_fasttext.keras",
    #"best_model_glove.keras",
    #"best_model_w2v.keras",
    "tokenizer_fasttext.json",
    #"tokenizer_glove.json",
    #"tokenizer_w2v.json"
]

for model in model_files:
    download_model_from_azure(model)
print("Tous les modèles sont prêts à être utilisés !")

# === Configuration des logs Azure Application Insights ===
connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if not connection_string:
    raise ValueError("APPLICATIONINSIGHTS_CONNECTION_STRING n'est pas défini dans les variables d'environnement.")

logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)
azure_handler = AzureLogHandler(connection_string=connection_string)
logger.addHandler(azure_handler)
logger.info("API démarrée - Logs connectés à Azure Application Insights")

# === Chargement du modèle ===
model_path = os.path.join(MODEL_DIR, "best_model_fasttext.keras")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modèle non trouvé : {model_path}")

try:
    model = tf.keras.models.load_model(model_path)
    logger.info("Modèle chargé avec succès !")
except Exception as e:
    logger.error(f"Erreur de chargement du modèle : {e}")
    raise ValueError(f"Erreur de chargement du modèle : {e}")

# === Chargement du tokenizer JSON ===
tokenizer_path = os.path.join(MODEL_DIR, "tokenizer_fasttext.json")
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer non trouvé : {tokenizer_path}")

try:
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_data)
    logger.info("Tokenizer chargé avec succès !")
except Exception as e:
    logger.error(f"Erreur de chargement du tokenizer : {e}")
    raise ValueError(f"Erreur de chargement du tokenizer : {e}")

# === Initialisation de FastAPI ===
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

@app.post("/predict")
async def predict(input: TextInput):
    try:
        if not input.text.strip():
            raise HTTPException(status_code=400, detail="Le texte d'entrée est vide")
        sequence = tokenizer.texts_to_sequences([input.text])
        sequence_padded = pad_sequences(sequence, maxlen=50, padding="post", truncating="post")
        prediction = model.predict(sequence_padded)
        sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
        logger.info(f"Prédiction : {sentiment} | Texte : {input.text}")
        return {"prediction": sentiment}
    except HTTPException as he:
        logger.warning(f"Mauvaise requête : {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Erreur de prédiction : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur serveur lors de la prédiction")

error_feedback_counter = {}

@app.post("/feedback")
async def feedback(input: FeedbackInput):
    try:
        if not input.text.strip():
            raise HTTPException(status_code=400, detail="Le texte du tweet est vide")
        if input.prediction not in ["positive", "negative"]:
            raise HTTPException(status_code=400, detail="Prédiction invalide")
        if not input.validation:
            logger.warning(f"Tweet mal prédit : {input.text} | Prédiction : {input.prediction}")
            error_feedback_counter[input.text] = error_feedback_counter.get(input.text, 0) + 1
            if error_feedback_counter[input.text] >= 3:
                logger.error("ALERTE : Plusieurs tweets mal prédits détectés !")
        return {"message": "Feedback enregistré, merci !"}
    except HTTPException as he:
        logger.warning(f"Mauvaise requête : {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Erreur lors du feedback : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur serveur lors du feedback")

@app.get("/test_log")
async def test_log():
    logger.info("Test de log envoyé à Application Insights.")
    return {"message": "Log envoyé avec succès à Azure !"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
