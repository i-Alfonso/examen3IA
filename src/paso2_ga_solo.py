import os as _os
import random
import time
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from utils.preprocesamiento import cargar_datos, clases_nombres
from utils.funcion_objetivo import evaluar
from utils.ga_local import ga_local

# ─── Colores ─────────────────────────────────────────────────────
R  = "\033[0m";  B  = "\033[1m"
AZ = "\033[94m"; CY = "\033[96m"; VE = "\033[92m"
RO = "\033[91m"; AM = "\033[93m"; GR = "\033[90m"

# ─── Datos ───────────────────────────────────────────────────────
X_train, X_val, X_test, y_train, y_val, y_test = cargar_datos()

# ─── Configuración del GA ────────────────────────────────────────
CONFIG = {
    "n_individuos"  : int(_os.environ.get("PARAM_N_INDIVIDUOS",   15)),
    "n_generaciones": int(_os.environ.get("PARAM_N_GENERACIONES", 20)),
    "p_cruce"       : float(_os.environ.get("PARAM_P_CRUCE",      0.8)),
    "p_mutacion"    : float(_os.environ.get("PARAM_P_MUTACION",   0.2)),
}

ACTIVACIONES = {0: "relu", 1: "tanh", 2: "logistic"}

print(f"{AZ}{B}{'='*55}{R}")
print(f"{AZ}{B}  PASO 2 — GA Solo{R}")
print(f"{AZ}{B}{'='*55}{R}")
print(f"\n{AM}Población   {R}: {B}{CONFIG['n_individuos']} individuos{R}")
print(f"{AM}Generaciones{R}: {B}{CONFIG['n_generaciones']}{R}")
print(f"{AM}MLPs totales{R}: {B}~{CONFIG['n_individuos'] * CONFIG['n_generaciones']}{R}")
print()

# ─── Ejecutar GA ─────────────────────────────────────────────────
random.seed(42)
t0 = time.time()

mejor, score_val, historial = ga_local(
    func_objetivo  = evaluar,
    centro         = None,
    radio          = 1.0,
    n_individuos   = CONFIG["n_individuos"],
    n_generaciones = CONFIG["n_generaciones"],
    p_cruce        = CONFIG["p_cruce"],
    p_mutacion     = CONFIG["p_mutacion"],
)

dt = time.time() - t0

# ─── Decodificar mejor individuo ─────────────────────────────────
mejor_lr        = mejor[0]
mejor_alpha     = mejor[1]
mejor_neuronas  = max(10,  int(round(mejor[2])))
mejor_capas     = max(1,   min(4, int(round(mejor[3]))))
mejor_activacion = ACTIVACIONES[int(round(mejor[4])) % 3]
mejor_max_iter  = max(100, min(500, int(round(mejor[5]))))

print(f"{AM}Tiempo total{R} : {B}{dt:.1f}s{R}")
print(f"\n{B}Mejores hiperparámetros encontrados:{R}")
print(f"  {AM}lr        {R} : {CY}{mejor_lr:.6f}{R}")
print(f"  {AM}alpha     {R} : {CY}{mejor_alpha:.8f}{R}")
print(f"  {AM}neuronas  {R} : {CY}{mejor_neuronas}{R}")
print(f"  {AM}capas     {R} : {CY}{mejor_capas}{R}  → {B}{tuple([mejor_neuronas]*mejor_capas)}{R}")
print(f"  {AM}activacion{R} : {CY}{mejor_activacion}{R}")
print(f"  {AM}max_iter  {R} : {CY}{mejor_max_iter}{R}")
print(f"\n{AM}Accuracy validación{R} : {CY}{B}{(1 - score_val)*100:.2f}%{R}")

# ─── Evaluar en test con MLP final ───────────────────────────────
modelo_final = MLPClassifier(
    hidden_layer_sizes  = tuple([mejor_neuronas] * mejor_capas),
    learning_rate_init  = mejor_lr,
    alpha               = mejor_alpha,
    activation          = mejor_activacion,
    max_iter            = mejor_max_iter,
    early_stopping      = True,
    validation_fraction = 0.1,
    n_iter_no_change    = 10,
    random_state        = 42,
)
modelo_final.fit(X_train, y_train)
acc_test = accuracy_score(y_test, modelo_final.predict(X_test))
print(f"{AM}Accuracy test      {R} : {VE}{B}{acc_test*100:.2f}%{R}")

# ─── Gráfica 1: evolución del GA ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
accuracy_hist = [1 - s for s in historial]
ax.plot(range(len(historial)), accuracy_hist, marker="o", color="steelblue", linewidth=2)
ax.set_xlabel("Generación")
ax.set_ylabel("Accuracy (validación)")
ax.set_title("Evolución del GA — Mejor accuracy por generación")
ax.set_ylim(0.5, 1.01)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graficas/ga/convergencia_ga.png", dpi=150)
plt.close()
print(f"\n{GR}Gráfica guardada: graficas/ga/convergencia_ga.png{R}")

# ─── Gráfica 2: matriz de confusión ──────────────────────────────
cm = confusion_matrix(y_test, modelo_final.predict(X_test))
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases_nombres()).plot(
    ax=ax, colorbar=True, cmap="Blues"
)
ax.set_title("Matriz de Confusión — GA Solo (Paso 2)\nAvila Bible")
plt.tight_layout()
plt.savefig("graficas/ga/matriz_confusion_ga.png", dpi=150)
plt.close()
print(f"{GR}Gráfica guardada: graficas/ga/matriz_confusion_ga.png{R}")

# ─── Tabla comparativa ───────────────────────────────────────────
print(f"\n{AZ}{'─'*55}{R}")
print(f"{B}  TABLA COMPARATIVA{R}")
print(f"{AZ}{'─'*55}{R}")
print(f"  {AM}MLP Base   {R} : {B}78.12%{R}  (hiperparámetros fijos)")
print(f"  {AM}GA Solo    {R} : {VE}{B}{acc_test*100:.2f}%{R}  (optimizados por GA)")
print(f"  {AM}WolfGenetic{R} : {GR}pendiente{R}")
