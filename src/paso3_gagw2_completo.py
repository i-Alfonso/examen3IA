import os as _os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from utils.preprocesamiento import cargar_datos, clases_nombres
from utils.funcion_objetivo import evaluar
from utils.gwo import gwo

# ─── Colores ─────────────────────────────────────────────────────
R  = "\033[0m";  B  = "\033[1m"
AZ = "\033[94m"; CY = "\033[96m"; VE = "\033[92m"
RO = "\033[91m"; AM = "\033[93m"; GR = "\033[90m"

# ─── Datos ───────────────────────────────────────────────────────
X_train, X_val, X_test, y_train, y_val, y_test = cargar_datos()

ACTIVACIONES = {0: "relu", 1: "tanh", 2: "logistic"}

# ─── Configuración WolfGenetic ───────────────────────────────────
CONFIG = {
    "n_lobos"    : int(_os.environ.get("PARAM_N_LOBOS",      10)),
    "max_iter"   : int(_os.environ.get("PARAM_MAX_ITER",     30)),
    "r_max"      : float(_os.environ.get("PARAM_R_MAX",      0.3)),
    "n_local"    : int(_os.environ.get("PARAM_N_INDIVIDUOS",   8)),
    "gen_locales": int(_os.environ.get("PARAM_N_GENERACIONES", 3)),
    "p_cruce"    : float(_os.environ.get("PARAM_P_CRUCE",    0.8)),
    "p_mutacion" : float(_os.environ.get("PARAM_P_MUTACION", 0.2)),
}

print(f"{AZ}{B}{'='*60}{R}")
print(f"{AZ}{B}  PASO 3 — WolfGenetic Completo{R}")
print(f"{AZ}{B}{'='*60}{R}")
print(f"\n{AM}Lobos      {R} : {B}{CONFIG['n_lobos']}{R}")
print(f"{AM}Iteraciones{R} : {B}{CONFIG['max_iter']}{R}")
print(f"{AM}GA local   {R} : {B}{CONFIG['n_local']} individuos × {CONFIG['gen_locales']} generaciones{R}")
print(f"{AM}MLPs totales{R}: {B}~{CONFIG['n_lobos'] * CONFIG['max_iter'] * CONFIG['n_local'] * CONFIG['gen_locales']}{R}")
print()

# ─── Ejecutar WolfGenetic ────────────────────────────────────────
random.seed(42)
t0 = time.time()

alpha, score_val, historial = gwo(
    func_objetivo = evaluar,
    n_lobos       = CONFIG["n_lobos"],
    max_iter      = CONFIG["max_iter"],
    r_max         = CONFIG["r_max"],
    n_local       = CONFIG["n_local"],
    gen_locales   = CONFIG["gen_locales"],
    p_cruce       = CONFIG["p_cruce"],
    p_mutacion    = CONFIG["p_mutacion"],
    verbose       = True,
)

dt = time.time() - t0

# ─── Decodificar alpha ───────────────────────────────────────────
mejor_lr       = alpha[0]
mejor_alpha    = alpha[1]
mejor_neuronas = max(10,  int(round(alpha[2])))
mejor_capas    = max(1,   min(4, int(round(alpha[3]))))
mejor_act      = ACTIVACIONES[int(round(alpha[4])) % 3]
mejor_max_iter = max(100, min(500, int(round(alpha[5]))))

print(f"\n{AM}Tiempo total{R} : {B}{dt:.1f}s  ({dt/60:.1f} min){R}")
print(f"\n{B}Mejores hiperparámetros (alpha):{R}")
print(f"  {AM}lr        {R} : {CY}{mejor_lr:.6f}{R}")
print(f"  {AM}alpha reg {R} : {CY}{mejor_alpha:.8f}{R}")
print(f"  {AM}neuronas  {R} : {CY}{mejor_neuronas}{R}")
print(f"  {AM}capas     {R} : {CY}{mejor_capas}{R}  → {B}{tuple([mejor_neuronas]*mejor_capas)}{R}")
print(f"  {AM}activacion{R} : {CY}{mejor_act}{R}")
print(f"  {AM}max_iter  {R} : {CY}{mejor_max_iter}{R}")
print(f"\n{AM}Accuracy validación{R} : {CY}{B}{(1-score_val)*100:.2f}%{R}")

# ─── Evaluar en test ─────────────────────────────────────────────
modelo_final = MLPClassifier(
    hidden_layer_sizes  = tuple([mejor_neuronas] * mejor_capas),
    learning_rate_init  = mejor_lr,
    alpha               = mejor_alpha,
    activation          = mejor_act,
    max_iter            = mejor_max_iter,
    early_stopping      = True,
    validation_fraction = 0.1,
    n_iter_no_change    = 10,
    random_state        = 42,
)
modelo_final.fit(X_train, y_train)
acc_test = accuracy_score(y_test, modelo_final.predict(X_test))
print(f"{AM}Accuracy test      {R} : {VE}{B}{acc_test*100:.2f}%{R}")

# ─── Gráfica 1: convergencia del alpha ───────────────────────────
iters     = list(range(1, len(historial["alpha_score"]) + 1))
acc_alpha = [1 - s for s in historial["alpha_score"]]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(iters, acc_alpha, marker="o", color="darkblue", linewidth=2, label="Alpha (mejor)")
ax.set_xlabel("Iteración GWO")
ax.set_ylabel("Accuracy (validación)")
ax.set_title("Convergencia del Alpha — WolfGenetic")
ax.set_ylim(max(0.5, min(acc_alpha) - 0.05), 1.01)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graficas/gagw2/convergencia_gagw2.png", dpi=150)
plt.close()
print(f"\n{GR}Gráfica 1 guardada: graficas/gagw2/convergencia_gagw2.png{R}")

# ─── Gráfica 2: radio por iteración ──────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(iters, historial["radio"], marker="s", color="steelblue", linewidth=2)
ax.set_xlabel("Iteración GWO")
ax.set_ylabel("Radio del GA local")
ax.set_title("Contracción del radio del GA local por iteración")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graficas/gagw2/radio_por_iteracion.png", dpi=150)
plt.close()
print(f"{GR}Gráfica 2 guardada: graficas/gagw2/radio_por_iteracion.png{R}")

# ─── Gráfica 3: boxplot de scores por iteración ──────────────────
acc_lobos = [[1 - s for s in scores] for scores in historial["scores_lobos"]]

fig, ax = plt.subplots(figsize=(14, 5))
ax.boxplot(acc_lobos, positions=iters, widths=0.6,
           patch_artist=True,
           boxprops=dict(facecolor="lightsteelblue", color="navy"),
           medianprops=dict(color="red", linewidth=2))
ax.set_xlabel("Iteración GWO")
ax.set_ylabel("Accuracy de los lobos (validación)")
ax.set_title("Distribución de accuracy de la manada por iteración")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("graficas/gagw2/boxplot_lobos.png", dpi=150)
plt.close()
print(f"{GR}Gráfica 3 guardada: graficas/gagw2/boxplot_lobos.png{R}")

# ─── Gráfica 4: tendencia central ────────────────────────────────
acc_media   = [1 - s for s in historial["media"]]
acc_mediana = [1 - s for s in historial["mediana"]]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(iters, acc_alpha,   marker="o", color="darkblue",  linewidth=2, label="Alpha (mejor)")
ax.plot(iters, acc_media,   marker="^", color="steelblue", linewidth=2, label="Media manada")
ax.plot(iters, acc_mediana, marker="s", color="tomato",    linewidth=2, label="Mediana manada")
ax.set_xlabel("Iteración GWO")
ax.set_ylabel("Accuracy (validación)")
ax.set_title("Tendencia central de la manada por iteración")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graficas/gagw2/tendencia_central.png", dpi=150)
plt.close()
print(f"{GR}Gráfica 4 guardada: graficas/gagw2/tendencia_central.png{R}")

# ─── Gráfica 5: matriz de confusión final ────────────────────────
cm = confusion_matrix(y_test, modelo_final.predict(X_test))
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases_nombres()).plot(
    ax=ax, colorbar=True, cmap="Blues"
)
ax.set_title("Matriz de Confusión — WolfGenetic (Paso 3)\nSensorless Drive")
plt.tight_layout()
plt.savefig("graficas/gagw2/matriz_confusion_final.png", dpi=150)
plt.close()
print(f"{GR}Gráfica 5 guardada: graficas/gagw2/matriz_confusion_final.png{R}")

# ─── Tabla comparativa final ─────────────────────────────────────
print(f"\n{AZ}{'='*60}{R}")
print(f"{B}  TABLA COMPARATIVA FINAL{R}")
print(f"{AZ}{'='*60}{R}")
print(f"  {AM}MLP Base   {R} : {B}??%{R}  (hiperparámetros fijos)")
print(f"  {AM}GA Solo    {R} : {B}??%{R}  (optimizados por GA)")
print(f"  {AM}WolfGenetic{R} : {VE}{B}{acc_test*100:.2f}%{R}  (optimizados por WolfGenetic)")
