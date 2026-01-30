import os
import re
import requests
from io import BytesIO
import joblib
from google import genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------
# CONFIGURACIÓN APP
# -----------------------------
app = FastAPI(
    title="API Sentimientos Pro v2",
    description="ML (Logistic Regression) + Gemini AI Contextual",
    version="2.0"
)

# -----------------------------
# STOPWORDS Y PALABRAS CLAVE
# -----------------------------
STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "y", "o", "de", "del", "al", "a", "en", "por", "para",
    "me", "te", "se", "mi", "mis", "tu", "tus",
    "es", "esta", "está", "estoy", "son", "era",
    "muy", "ya", "pero", "si"
}

PALABRAS_CLAVE = {
    "frustra", "frustrado", "frustrante", "lento", "lenta", "malo", "mala",
    "peor", "basura", "horrible", "terrible", "odio", "fallo", "error",
    "excelente", "bueno", "buena", "genial", "increible", "amo", "encanta",
    "gracias", "mejor", "perfecto", "no", "nunca", "jamas"
}

def limpiar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto)
    palabras = texto.split()
    palabras_limpias = [
        p for p in palabras 
        if (p not in STOPWORDS_ES or p in PALABRAS_CLAVE) and (len(p) > 2 or p == "no")
    ]
    return " ".join(palabras_limpias)

# -----------------------------
# CARGA MODELOS DESDE GOOGLE DRIVE
# -----------------------------
# Reemplaza estos IDs con los tuyos
URL_MODELO_SENTIMIENTOS = "https://drive.google.com/uc?export=download&id=1x-TAd6dRvc3oQgfya6bx1D4FWWWniGME"
URL_VECTORIZER = "https://drive.google.com/uc?export=download&id=1vz7i9jK7l1aO4da3pbcesIkVRvHjr0Gl"

try:
    # Descargar modelo
    r_model = requests.get(URL_MODELO_SENTIMIENTOS)
    modelo_sentimientos = joblib.load(BytesIO(r_model.content))

    # Descargar vectorizador
    r_vect = requests.get(URL_VECTORIZER)
    vectorizer = joblib.load(BytesIO(r_vect.content))

    print("Modelo y Vectorizador cargados desde Google Drive con éxito")
except Exception as e:
    print(f"Error al cargar modelos desde Drive: {e}")
    modelo_sentimientos = None
    vectorizer = None

# -----------------------------
# GEMINI AI
# -----------------------------
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

# -----------------------------
# ESQUEMAS
# -----------------------------
class TextoEntrada(BaseModel):
    text: str

# -----------------------------
# ENDPOINTS
# -----------------------------
@app.get("/")
async def root():
    return {"message": "API de Sentimientos Operativa"}

@app.post("/predict")
async def predict(data: TextoEntrada):
    if modelo_sentimientos is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible.")

    if client is None:
        raise HTTPException(status_code=500, detail="Configuración de IA faltante.")

    texto_limpio = limpiar_texto(data.text)

    # Predicción ML
    try:
        texto_tfidf = vectorizer.transform([texto_limpio])
        prediccion = modelo_sentimientos.predict(texto_tfidf)[0]
        probas = modelo_sentimientos.predict_proba(texto_tfidf)[0]
        clases = modelo_sentimientos.classes_
        idx_pred = list(clases).index(prediccion)
        confianza = probas[idx_pred]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error en predicción: {e}")

    # Mensaje Gemini
    prompt = f"""
    Actúa como un asistente juvenil, cercano y empático.
    El usuario escribió: "{data.text}"
    El modelo detectó un sentimiento: {prediccion}.
    Genera una respuesta muy breve (máximo 2 frases) que sea coherente con lo que el usuario expresó.
    Usa emojis.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        mensaje_gemini = response.text.strip()
    except Exception:
        if prediccion == "positive":
            mensaje_gemini = "¡Qué alegría leer eso! ✨ Muchas gracias por tu comentario."
        else:
            mensaje_gemini = "Lamento mucho esa experiencia. 😔 Ya estamos trabajando para mejorarlo."

    return {
        "sentimiento": prediccion,
        "confianza": round(float(confianza), 4),
        "mensaje_gemini": mensaje_gemini
    }
