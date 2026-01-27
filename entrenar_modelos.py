import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Importar datasets
frutas = pd.read_csv("datasets/frutas.csv")
deportes = pd.read_csv("datasets/deportes.csv")

# Separar etiquetas
X_frutas = frutas.drop("fruta", axis=1)
y_frutas = frutas["fruta"]

X_deportes = deportes.drop("deporte", axis=1)
y_deportes = deportes["deporte"]

# Datos de entrenamiento y prueba (80% / 20%)
Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_frutas, y_frutas, test_size=0.2, random_state=42)
Xd_train, Xd_test, yd_train, yd_test = train_test_split(X_deportes, y_deportes, test_size=0.2, random_state=42)

# Crear y entrenar modelos
modelo_frutas = DecisionTreeClassifier(random_state=42)
modelo_frutas.fit(Xf_train, yf_train)

modelo_deportes = DecisionTreeClassifier(random_state=42)
modelo_deportes.fit(Xd_train, yd_train)

# Predicciones
yf_pred = modelo_frutas.predict(Xf_test)
yd_pred = modelo_deportes.predict(Xd_test)

print("=== MÉTRICAS DEL MODELO DE FRUTAS ===")
# Exactitud
exactitud = accuracy_score(yf_test, yf_pred)
print("Exactidud:", exactitud)

# Matriz de confusion
Matriz_confusion = confusion_matrix(yf_test, yf_pred)
print("Matriz de Confusión:\n", Matriz_confusion)

# Heatmap matriz de confusión - frutas
labels_frutas = sorted(list(set(yf_test) | set(yf_pred)))
matriz_confusion_frutas = pd.DataFrame(Matriz_confusion, index=labels_frutas, columns=labels_frutas)
sns.heatmap(matriz_confusion_frutas, annot=True, fmt='d', cmap='viridis')
plt.title("Matriz de Confusión - Frutas")
plt.ylabel("Etiqueta real")
plt.xlabel("Predicción")
plt.show()

# Reporte de clasificacion
reporte_clasificacion = classification_report(yf_test, yf_pred)
print("Reporte de clasificacion:\n", reporte_clasificacion)


print("\n=== MÉTRICAS DEL MODELO DE DEPORTES ===")
# Exactitud
exactitudD = accuracy_score(yd_test, yd_pred)
print("Exactidud:", exactitudD)

# Matriz de confusion
Matriz_confusionD = confusion_matrix(yd_test, yd_pred)
print("Matriz de Confusión:\n", Matriz_confusionD)

# Heatmap matriz de confusión - deportes
labels_deportes = sorted(list(set(yd_test) | set(yd_pred)))
matriz_deportes = pd.DataFrame(Matriz_confusionD, index=labels_deportes, columns=labels_deportes)
sns.heatmap(matriz_deportes, annot=True, fmt='d', cmap='viridis')
plt.title("Matriz de Confusión - Deportes")
plt.ylabel("Etiqueta real")
plt.xlabel("Predicción")
plt.show()

# Reporte de clasificacion
reporte_clasificacionD = classification_report(yd_test, yd_pred)
print("Reporte de clasificacion:\n", reporte_clasificacionD)

# Guardar modelos entrenados
joblib.dump(modelo_frutas, "modelo_frutas.pkl")
joblib.dump(modelo_deportes, "modelo_deportes.pkl")

print("\nModelos (frutas y deportes) guardados correctamente")