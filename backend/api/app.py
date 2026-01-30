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
URL_MODELO = "https://drive.google.com/uc?export=download&id=1dgL6VqbuTcIsSW-Co5Y2v_BxdDYkQOS7"
URL_VECTORIZER = "https://drive.google.com/uc?export=download&id=1lGMt17wNkflyzKnDbwVOmSxUGyCnBP7j"

def limpiar_texto(texto: str) -> str:
    # 1. Minúsculas y limpieza básica
    texto = texto.lower()
    
    # 2. Remover acentos para normalizar (ej: 'gustó' -> 'gusto')
    import unicodedata
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn')

    # 3. Mantener solo letras
    texto = re.sub(r"[^a-zñ\s]", "", texto)

    # 4. Lista de palabras ruidosas (Stop Words)
    # Eliminamos artículos, preposiciones y pronombres que no aportan sentimiento
    stop_words = {
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del', 
        'a', 'ante', 'con', 'por', 'para', 'su', 'sus', 'tu', 'tus', 'esto', 
        'esta', 'este', 'ese', 'esa', 'estos', 'estas', 'que', 'en', 'y'
    }
    
    palabras = texto.split()
    palabras_limpias = [p for p in palabras if p not in stop_words]
    
    return " ".join(palabras_limpias).strip()

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
    # 1. Validación de API Key y Carga de modelos (se mantiene igual)
    api_key = os.environ.get("GEMINI_API_KEY")
    try:
        cargar_modelos()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Predicción con Lógica de Calibración (IGUAL A TU ENTRENAMIENTO)
    try:
        texto_limpio = limpiar_texto(data.text)
        tfidf = state.vectorizer.transform([texto_limpio])
        
        # Obtenemos las probabilidades
        prob = state.modelo_sentimientos.predict_proba(tfidf)[0]
        clases = state.modelo_sentimientos.classes_
        
        # Buscamos el índice de la clase positiva (ajusta si tu CSV usa '1' o 'pos')
        idx_pos = list(clases).index("positive") if "positive" in clases else 1
        prob_positive_percent = prob[idx_pos] * 100

        # Aplicamos tu lógica de rangos
        if prob_positive_percent > 52.5:
            pred_final = "positive"
        elif prob_positive_percent < 47.5:
            pred_final = "negative"
        else:
            pred_final = "neutral"
            
        confianza = float(max(prob))
    except Exception as e:
        print(f"Fallo en predicción ML: {e}")
        pred_final = "Indeterminado"
        confianza = 0.0

    # 4. Respuesta con Gemini (Prompt reforzado)
    try:
        client = genai.Client(api_key=api_key)
        
        # Le decimos a Gemini que NO ignore nuestra predicción
        prompt = (
            f"Contexto: Eres un amigo empático. Un sistema de IA analizó el mensaje del usuario "
            f"y determinó que el sentimiento es: {pred_final}. "
            f"El usuario dijo: '{data.text}'. "
            f"Tu tarea: Responde al usuario en una sola frase corta con emojis que coincida que esa minimo de 3 frases "
            f"estrictamente con el sentimiento '{pred_final}' detectado."
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )
        mensaje = response.text.strip() if response.text else "¡Te entiendo! ✨"
        
    except Exception as e:
        mensaje = "¡Te entiendo perfectamente! Aquí estoy para ti. ✨"

    return {
        "sentimiento": pred_final,
        "confianza": round(confianza, 2),
        "mensaje_gemini": mensaje
    }