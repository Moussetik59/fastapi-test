import streamlit as st
import requests

# URL de l'API FastAPI **locale**
# API_URL = "http://localhost:8000"
API_URL = "https://fastapi-test-app-gcbtd0a4eehufcfw.northeurope-01.azurewebsites.net"

# Titre et présentation
st.title("📝 Analyse de Sentiments")
st.write("Entrez un texte et obtenez une prédiction de sentiment (positif ou négatif)")

# Entrée utilisateur
user_input = st.text_area("Texte à analyser", "")

# Initialiser `session_state` pour stocker la prédiction
if "result" not in st.session_state:
    st.session_state.result = None

if st.button("Analyser"):
    if user_input:
        with st.spinner("🔍 Analyse en cours..."):
            try:
                # Envoie la requête à l'API FastAPI
                response = requests.post(f"{API_URL}/predict", json={"text": user_input})
                st.session_state.result = response.json()  # Stocker le résultat dans session_state

                if "prediction" in st.session_state.result:
                    sentiment = "😊 Positif" if st.session_state.result["prediction"] == "positive" else "😡 Négatif"
                    st.success(f"Résultat : {sentiment}")
                else:
                    st.error("❌ Erreur lors de l'analyse. Vérifiez l'API.")
            except Exception as e:
                st.error(f"⚠️ Erreur : {e}")
    else:
        st.warning("🚨 Entrez un texte avant d'envoyer !")

# Section Feedback
st.subheader("📩 Feedback sur la Prédiction")
prediction_feedback = st.radio("La prédiction était-elle correcte ?", ("Oui", "Non"))

if st.button("Envoyer Feedback"):
    if user_input and st.session_state.result is not None and "prediction" in st.session_state.result:  # Vérification avec session_state
        feedback_data = {
            "text": user_input,
            "prediction": st.session_state.result["prediction"],
            "validation": True if prediction_feedback == "Oui" else False,
        }

        try:
            response = requests.post(f"{API_URL}/feedback", json=feedback_data)
            if response.status_code == 200:
                st.success("✅ Feedback envoyé ! Merci 😊")
            else:
                st.error(f"⚠️ Erreur lors de l'envoi du feedback : {response.text}")
        except Exception as e:
            st.error(f"⚠️ Erreur lors de l'envoi du feedback : {e}")
    else:
        st.warning("🚨 Vous devez d'abord analyser un texte avant de donner un feedback !")
