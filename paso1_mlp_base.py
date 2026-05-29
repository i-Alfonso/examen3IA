import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from utils.preprocesamiento import cargar_datos

# ─── Datos ───────────────────────────────────────────────────────
X_train, X_val, X_test, y_train, y_val, y_test = cargar_datos()

# ─── Hiperparámetros fijos (baseline, sin optimizar) ─────────────
hiperparametros = {
    "hidden_layer_sizes" : (100, 100),
    "activation"         : "relu",
    "learning_rate_init" : 0.001,
    "alpha"              : 0.0001,
    "max_iter"           : 300,
    "early_stopping"     : True,
    "validation_fraction": 0.1,
    "n_iter_no_change"   : 10,
    "random_state"       : 42,
}

# ─── Entrenar ────────────────────────────────────────────────────
print("=" * 50)
print("  PASO 1 — MLP Base (hiperparámetros fijos)")
print("=" * 50)
print(f"\nArquitectura : {hiperparametros['hidden_layer_sizes']}")
print(f"Activación   : {hiperparametros['activation']}")
print(f"LR           : {hiperparametros['learning_rate_init']}")
print(f"Alpha        : {hiperparametros['alpha']}")
print()

modelo = MLPClassifier(**hiperparametros)

t0 = time.time()
modelo.fit(X_train, y_train)
dt = time.time() - t0

# ─── Evaluar ─────────────────────────────────────────────────────
acc_val  = accuracy_score(y_val,  modelo.predict(X_val))
acc_test = accuracy_score(y_test, modelo.predict(X_test))

print(f"Épocas entrenadas   : {modelo.n_iter_}")
print(f"Tiempo              : {dt:.2f}s")
print(f"Accuracy validación : {acc_val:.4f}  ({acc_val*100:.2f}%)")
print(f"Accuracy test       : {acc_test:.4f}  ({acc_test*100:.2f}%)")
print()
print("Criterio de aceptación: accuracy >= 0.93")
print(f"  {'PASA' if acc_val >= 0.93 else 'FALLA'}")

# ─── Matriz de confusión ─────────────────────────────────────────
cm = confusion_matrix(y_test, modelo.predict(X_test))
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title("Matriz de Confusión — MLP Base (Paso 1)\nPendigits UCI")
plt.tight_layout()
plt.savefig("graficas/matriz_confusion_base.png", dpi=150)
plt.close()
print("\nGráfica guardada: graficas/matriz_confusion_base.png")

# ─── Resumen para la tabla comparativa final ─────────────────────
print()
print("─" * 50)
print("  RESUMEN (para tabla comparativa del reporte)")
print("─" * 50)
print(f"  Modelo          : MLP Base")
print(f"  Hiperparámetros : fijos manualmente")
print(f"  Acc. validación : {acc_val*100:.2f}%")
print(f"  Acc. test       : {acc_test*100:.2f}%")
