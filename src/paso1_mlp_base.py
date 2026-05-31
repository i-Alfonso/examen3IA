import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from utils.preprocesamiento import cargar_datos, clases_nombres

# ─── Colores ─────────────────────────────────────────────────────
R  = "\033[0m";  B  = "\033[1m"
AZ = "\033[94m"; CY = "\033[96m"; VE = "\033[92m"
RO = "\033[91m"; AM = "\033[93m"; GR = "\033[90m"

# ─── Datos ───────────────────────────────────────────────────────
X_train, X_val, X_test, y_train, y_val, y_test = cargar_datos()

# ─── Leer parámetros (del menú o valores por defecto) ────────────
_neuronas = int(os.environ.get("PARAM_HIDDEN_LAYERS", 100))
_capas    = int(os.environ.get("PARAM_N_CAPAS",       2))
_act      = os.environ.get("PARAM_ACTIVATION",        "relu")
_lr       = float(os.environ.get("PARAM_LR",          0.001))
_alpha    = float(os.environ.get("PARAM_ALPHA",        0.0001))
_max_iter = int(os.environ.get("PARAM_MAX_ITER",       300))

hiperparametros = {
    "hidden_layer_sizes" : tuple([_neuronas] * _capas),
    "activation"         : _act,
    "learning_rate_init" : _lr,
    "alpha"              : _alpha,
    "max_iter"           : _max_iter,
    "early_stopping"     : True,
    "validation_fraction": 0.1,
    "n_iter_no_change"   : 10,
    "random_state"       : 42,
}

# ─── Entrenar ────────────────────────────────────────────────────
print(f"{AZ}{B}{'='*50}{R}")
print(f"{AZ}{B}  PASO 1 — MLP Base{R}")
print(f"{AZ}{B}{'='*50}{R}")
print(f"\n{AM}Arquitectura{R} : {B}{hiperparametros['hidden_layer_sizes']}{R}")
print(f"{AM}Activación  {R} : {B}{hiperparametros['activation']}{R}")
print(f"{AM}LR          {R} : {B}{hiperparametros['learning_rate_init']}{R}")
print(f"{AM}Alpha       {R} : {B}{hiperparametros['alpha']}{R}")
print()

modelo = MLPClassifier(**hiperparametros)
t0 = time.time()
modelo.fit(X_train, y_train)
dt = time.time() - t0

# ─── Evaluar ─────────────────────────────────────────────────────
acc_val  = accuracy_score(y_val,  modelo.predict(X_val))
acc_test = accuracy_score(y_test, modelo.predict(X_test))

print(f"{AM}Épocas entrenadas  {R} : {B}{modelo.n_iter_}{R}")
print(f"{AM}Tiempo             {R} : {B}{dt:.2f}s{R}")
print(f"{AM}Accuracy validación{R} : {CY}{B}{acc_val*100:.2f}%{R}")
print(f"{AM}Accuracy test      {R} : {VE}{B}{acc_test*100:.2f}%{R}")
print()

pasa = acc_val >= 0.75
estado = f"{VE}{B}PASA{R}" if pasa else f"{RO}{B}FALLA{R}"
print(f"Criterio (accuracy ≥ 0.75) : {estado}")

# ─── Matriz de confusión ─────────────────────────────────────────
cm = confusion_matrix(y_test, modelo.predict(X_test))
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases_nombres()).plot(
    ax=ax, colorbar=True, cmap="Blues"
)
ax.set_title("Matriz de Confusión — MLP Base (Paso 1)\nAvila Bible")
plt.tight_layout()
plt.savefig("graficas/mlp_base/matriz_confusion_base.png", dpi=150)
plt.close()
print(f"\n{GR}Gráfica guardada: graficas/mlp_base/matriz_confusion_base.png{R}")

# ─── Resumen ─────────────────────────────────────────────────────
print(f"\n{AZ}{'─'*50}{R}")
print(f"{B}  RESUMEN — Tabla comparativa{R}")
print(f"{AZ}{'─'*50}{R}")
print(f"  {AM}Modelo         {R} : MLP Base")
print(f"  {AM}Hiperparámetros{R} : fijos manualmente")
print(f"  {AM}Acc. validación{R} : {CY}{B}{acc_val*100:.2f}%{R}")
print(f"  {AM}Acc. test      {R} : {VE}{B}{acc_test*100:.2f}%{R}")
