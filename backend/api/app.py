import os
import re
import requests
import joblib
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <--- ESTO FALTABA
from pydantic import BaseModel
from google import genai
load_dotenv()
print(f"La clave cargada empieza con: {os.environ.get('GEMINI_API_KEY')[:5]}...")
app = FastAPI()

# Configuración de CORS
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://tu-frontend.vercel.app",
    "*" # Permite acceso total mientras testeas
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RECURSOS GLOBALES ---
# Usamos una clase simple para mantener el estado del modelo en memoria caché
class ModelState:
    modelo_sentimientos = None
    vectorizer = None

state = ModelState()

# IDs de Google Drive
URL_MODELO = "https://drive.google.com/uc?export=download&id=1x-TAd6dRvc3oQgfya6bx1D4FWWWniGME"
URL_VECTORIZER = "https://drive.google.com/uc?export=download&id=1vz7i9jK7l1aO4da3pbcesIkVRvHjr0Gl"

# --- UTILIDADES ---
STOPWORDS_ES = {"el", "la", "los", "las", "un", "una", "y", "o", "de", "en", "por", "a"}
PALABRAS_CLAVE = {"no", "lento", "malo", "excelente", "bueno", "error", "fallo"}

def limpiar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto)
    palabras = texto.split()
    return " ".join([p for p in palabras if (p not in STOPWORDS_ES or p in PALABRAS_CLAVE)])

def cargar_modelos_si_no_existen():
    if state.modelo_sentimientos is None or state.vectorizer is None:
        try:
            session = requests.Session()
            def download_drive_file(url):
                response = session.get(url, stream=True, timeout=20)
                # Si Google pide confirmación de virus, extraemos el token
                token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        token = value
                if token:
                    response = session.get(url, params={'confirm': token}, stream=True)
                return response.content

            print("Descargando modelos...")
            content_m = download_drive_file(URL_MODELO)
            state.modelo_sentimientos = joblib.load(BytesIO(content_m))
            
            content_v = download_drive_file(URL_VECTORIZER)
            state.vectorizer = joblib.load(BytesIO(content_v))
            print("Modelos cargados.")
        except Exception as e:
            # Esto te dirá exactamente qué falló en los logs
            print(f"ERROR CARGANDO MODELOS: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Carga fallida: {str(e)}")
# --- ENDPOINTS ---
class TextoEntrada(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "ok", "message": "API de Sentimientos Operativa"}

@app.post("/predict")
async def predict(data: TextoEntrada):
    # 1. Verificar API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Falta GEMINI_API_KEY en variables de entorno.")

    # 2. Cargar modelos (solo si no están en memoria)
    cargar_modelos_si_no_existen()

    # 3. Preprocesamiento y Predicción ML
    try:
        texto_limpio = limpiar_texto(data.text)
        tfidf = state.vectorizer.transform([texto_limpio])
        pred = state.modelo_sentimientos.predict(tfidf)[0]
        prob = state.modelo_sentimientos.predict_proba(tfidf)[0]
        confianza = float(max(prob))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error en ML: {str(e)}")

    # 4. Generación con Gemini AI
    try:
        client = genai.Client(api_key=api_key)
        
        # Nuevo prompt integrado
        prompt = f"""
        Actúa como un asistente juvenil, cercano y empático.
        El usuario escribió: "{data.text}"
        El modelo detectó un sentimiento: {pred}.
        Genera una respuesta muy breve (máximo 2 frases) que sea coherente con lo que el usuario expresó.
        Usa emojis.
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )
        mensaje = response.text.strip()
    except Exception as e:
        print(f"Error en Gemini: {e}")
        mensaje = "¡Gracias por compartirlo! ❤️" if pred == "positive" else "Tranqui, aquí estoy para lo que necesites. 😔"
    return {
        "sentimiento": pred,
        "confianza": round(confianza, 4),
        "mensaje_gemini": mensaje
    }