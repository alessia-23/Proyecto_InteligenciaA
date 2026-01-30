import os
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------------
# 1. Cargar dataset
# -----------------------------------
sentimientos = pd.read_csv("sentiment.csv")

# Verificar columnas necesarias
if "text" not in sentimientos.columns or "sentiment" not in sentimientos.columns:
    raise ValueError("❌ El CSV debe tener las columnas: text y sentiment")

X = sentimientos["text"]
y = sentimientos["sentiment"]

# -----------------------------------
# 2. Ver distribución de clases
# -----------------------------------
print("📌 Distribución de clases:")
print(y.value_counts())

if y.nunique() < 2:
    raise ValueError("❌ El dataset debe tener al menos 2 clases")

# -----------------------------------
# 3. División de datos (80% / 20%)
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------
# 4. Vectorización TF-IDF
# -----------------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------------
# 5. Entrenamiento del modelo
# -----------------------------------
modelo_sentimientos = LogisticRegression(max_iter=1000, random_state=42)
modelo_sentimientos.fit(X_train_tfidf, y_train)

# -----------------------------------
# 6. Predicciones
# -----------------------------------
y_pred = modelo_sentimientos.predict(X_test_tfidf)
y_proba = modelo_sentimientos.predict_proba(X_test_tfidf)

# -----------------------------------
# 6b. Calibrar niveles de confianza según tabla de polaridad
# -----------------------------------
clases = modelo_sentimientos.classes_
if "positive" in clases:
    idx_pos = list(clases).index("positive")
else:
    # Ajusta si tus clases son diferentes
    idx_pos = 1

prob_positive_percent = y_proba[:, idx_pos] * 100

def asignar_categoria(prob_percent):
    if prob_percent > 52.5:
        return "Positivo"
    elif prob_percent < 47.5:
        return "Negativo"
    else:
        return "Neutro"

categorias = [asignar_categoria(p) for p in prob_positive_percent]

# Ejemplos de verificación
print("\n=== EJEMPLOS DE CATEGORIZACIÓN ===")
for i in range(min(10, len(X_test))):
    print(f"Texto: {X_test.iloc[i]}")
    print(f"Predicción: {y_pred[i]}, Prob. positivo %: {prob_positive_percent[i]:.2f}, Categoría: {categorias[i]}")
    print("---")

# -----------------------------------
# 7. Métricas
# -----------------------------------
print("\n=== MÉTRICAS DEL MODELO DE SENTIMIENTOS ===")
exactitud = accuracy_score(y_test, y_pred)
print("Exactitud:", exactitud)

matriz_confusion = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusión:")
print(matriz_confusion)

labels = sorted(list(set(y_test) | set(y_pred)))
matriz_df = pd.DataFrame(
    matriz_confusion,
    index=labels,
    columns=labels
)

sns.heatmap(matriz_df, annot=True, fmt="d", cmap="viridis")
plt.title("Matriz de Confusión - Análisis de Sentimientos")
plt.ylabel("Etiqueta real")
plt.xlabel("Predicción")
plt.show()

reporte = classification_report(y_test, y_pred)
print("\nReporte de Clasificación:")
print(reporte)

# -----------------------------------
# 8. Guardar modelo y vectorizador
# -----------------------------------
MODEL_PATH = os.path.join(os.getcwd(), "models")
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

try:
    joblib.dump(modelo_sentimientos, os.path.join(MODEL_PATH, "sentiment_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"))
    print("Guardando vectorizador en:", os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"))
    print("\n✅ Modelo de sentimientos y vectorizador guardados correctamente en 'models/'")
except Exception as e:
    print(f"❌ Error guardando archivos: {e}")

