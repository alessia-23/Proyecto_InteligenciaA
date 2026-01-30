import os
import re
import requests
import joblib
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

load_dotenv()

app = FastAPI()

# Configuración de CORS para que tu frontend en Vercel pueda conectar
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextoEntrada(BaseModel):
    text: str

class ModelState:
    modelo_sentimientos = None
    vectorizer = None

state = ModelState()

# IDs de Google Drive directos
URL_MODELO = "https://drive.google.com/uc?export=download&id=1x-TAd6dRvc3oQgfya6bx1D4FWWWniGME"
URL_VECTORIZER = "https://drive.google.com/uc?export=download&id=1vz7i9jK7l1aO4da3pbcesIkVRvHjr0Gl"

def limpiar_texto(texto: str) -> str:
    texto = texto.lower()
    # Mantenemos solo letras y espacios para reducir dimensiones
    texto = re.sub(r"[^a-záéíóúñ\s]", "", texto)
    return texto.strip()

def cargar_modelos():
    """Carga los modelos solo cuando se hace la primera petición (ahorra RAM al inicio)"""
    if state.modelo_sentimientos is None or state.vectorizer is None:
        try:
            print("Iniciando descarga de modelos desde Drive...")
            session = requests.Session()
            def download_file(url):
                response = session.get(url, stream=True, timeout=15)
                token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        token = value
                if token:
                    response = session.get(url, params={'confirm': token}, stream=True)
                return response.content

            # Carga en memoria
            state.modelo_sentimientos = joblib.load(BytesIO(download_file(URL_MODELO)))
            state.vectorizer = joblib.load(BytesIO(download_file(URL_VECTORIZER)))
            print("Modelos listos para usar.")
        except Exception as e:
            print(f"Error crítico cargando modelos: {e}")
            raise RuntimeError("No se pudieron cargar los modelos de ML.")

@app.get("/")
def home():
    return {"status": "conectado", "proyecto": "Akinator de Sentimientos - Ale"}

@app.post("/predict")
async def predict(data: TextoEntrada):
    # 1. Validación de API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Configura GEMINI_API_KEY en Vercel")

    # 2. Carga perezosa (Lazy Load)
    try:
        cargar_modelos()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Predicción con tu modelo entrenado
    try:
        texto_limpio = limpiar_texto(data.text)
        tfidf = state.vectorizer.transform([texto_limpio])
        pred = state.modelo_sentimientos.predict(tfidf)[0]
        
        # Intentar obtener probabilidad si el modelo lo permite
        confianza = 0.0
        if hasattr(state.modelo_sentimientos, "predict_proba"):
            prob = state.modelo_sentimientos.predict_proba(tfidf)[0]
            confianza = float(max(prob))
    except Exception as e:
        print(f"Fallo en predicción ML: {e}")
        pred = "indeterminado"
        confianza = 0.0

    # 4. Respuesta creativa con Gemini
    try:
        client = genai.Client(api_key=api_key)
        prompt = (
            f"Actúa como un amigo empático y buena onda. "
            f"El usuario dijo: '{data.text}'. "
            f"El sistema detectó que se siente: {pred}. "
            "Responde en una sola frase muy corta con emojis que valide su emoción."
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )
        mensaje = response.text.strip()
    except Exception as e:
        mensaje = "¡Te entiendo perfectamente! Aquí estoy para ti. ✨"

    return {
        "sentimiento": str(pred),
        "confianza": round(confianza, 2),
        "mensaje_gemini": mensaje
    }