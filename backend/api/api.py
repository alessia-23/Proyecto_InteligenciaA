import os
import re
import joblib
import pandas as pd
from google import genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------------
# CONFIG APP
# -----------------------------------
app = FastAPI(
    title="API Sentimientos Pro v2",
    description="ML (Logistic Regression) + Gemini AI Contextual",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# CONFIGURACIÓN DE PROCESAMIENTO
# -----------------------------------
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
    # Eliminar caracteres especiales pero mantener espacios
    texto = re.sub(r"[^\w\s]", " ", texto)
    palabras = texto.split()
    
    # REGLA ORO: No eliminar "no" y mantener palabras clave aunque sean cortas
    palabras_limpias = [
        p for p in palabras 
        if (p not in STOPWORDS_ES or p in PALABRAS_CLAVE) and (len(p) > 2 or p == "no")
    ]
    return " ".join(palabras_limpias)

# -----------------------------------
# CARGA DE MODELO Y IA
# -----------------------------------
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBZ98jRlyoVo2zCG_nnyhirhr96qigKGI0")
client = genai.Client(api_key=GEMINI_KEY)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "models")

try:
    modelo_sentimientos = joblib.load(os.path.join(MODEL_PATH, "sentiment_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"))
    print("Modelo y Vectorizador cargados con éxito")
except Exception as e:
    print(f"Error crítico al cargar modelos: {e}")
    modelo_sentimientos = None
    vectorizer = None

# -----------------------------------
# ESQUEMAS Y ENDPOINTS
# -----------------------------------
class TextoEntrada(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: TextoEntrada):
    if modelo_sentimientos is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="El modelo no se cargó correctamente.")

    # 1. Preprocesamiento
    texto_limpio = limpiar_texto(data.text)
    
    # 2. Predicción de ML
    try:
        texto_tfidf = vectorizer.transform([texto_limpio])
        prediccion = modelo_sentimientos.predict(texto_tfidf)[0]
        
        # Probabilidades para la confianza
        probas = modelo_sentimientos.predict_proba(texto_tfidf)[0]
        clases = modelo_sentimientos.classes_
        idx_pred = list(clases).index(prediccion)
        confianza = probas[idx_pred]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error en predicción: {e}")

    # 3. Generar mensaje con Gemini usando CONTEXTO REAL
    prompt = f"""
    Actúa como un asistente juvenil, cercano y empático.
    El usuario escribió: "{data.text}"
    El modelo detectó un sentimiento: {prediccion}.

    Genera una respuesta muy breve (máximo 2 frases) que sea coherente con lo que el usuario expresó:
    - Si es negativo (quejas de lentitud, fallos, etc.), ofrece apoyo y di que estamos trabajando para mejorar.
    - Si es positivo, agradece con entusiasmo.
    Usa emojis de forma natural. No uses negritas excesivas.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", # Usamos el modelo más rápido
            contents=prompt
        )
        mensaje_gemini = response.text.strip()
    except Exception:
        # Fallback si Gemini falla
        if prediccion == "positive":
            mensaje_gemini = "¡Qué alegría leer eso! ✨ Muchas gracias por tu comentario, nos motiva a seguir mejorando. 🚀"
        else:
            mensaje_gemini = "Lamento mucho que tengas esa experiencia. 😔 Ya estamos revisando qué pasó para arreglarlo pronto. ¡Gracias por avisarnos! 🛠️"

    return {
        "texto_original": data.text,
        "texto_limpio": texto_limpio,
        "sentimiento": prediccion,
        "confianza": round(float(confianza), 4),
        "mensaje_gemini": mensaje_gemini
    }