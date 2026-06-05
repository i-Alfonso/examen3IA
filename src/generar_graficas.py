"""
generar_graficas.py — Construye las gráficas a partir de los datos guardados

No entrena nada. Solo lee los JSON más recientes de cada simulación
(resultados/mlp_puro/ y resultados/wolfgenetic/) y dibuja las gráficas
que pide el parcial, guardándolas en graficas/.

Se ejecuta desde main.py o directamente:
    PYTHONPATH=. python3 src/generar_graficas.py
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

from utils.registro import ultimo_json

# Carpeta donde se guardan las gráficas
_GDIR = os.path.join(os.path.dirname(__file__), "..", "graficas")
os.makedirs(_GDIR, exist_ok=True)


def _guardar(nombre):
    ruta = os.path.join(_GDIR, nombre)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  guardada: graficas/{nombre}")


def _matriz_confusion(matriz, clases, titulo, archivo):
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=np.array(matriz),
                           display_labels=clases).plot(
        ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(titulo)
    _guardar(archivo)


def graficas_wolfgenetic(datos):
    """Genera las gráficas de evolución del WolfGenetic."""
    h = datos["historial"]
    iters     = list(range(1, len(h["alpha_score"]) + 1))
    acc_alpha = [1 - s for s in h["alpha_score"]]
    acc_media = [1 - s for s in h["media"]]
    acc_medi  = [1 - s for s in h["mediana"]]

    # 1. Convergencia del alfa por iteración (evolución del GWO)
    plt.figure(figsize=(9, 4))
    plt.plot(iters, acc_alpha, marker="o", color="darkblue", linewidth=2)
    plt.xlabel("Iteración GWO"); plt.ylabel("Accuracy (validación)")
    plt.title("Convergencia del Alfa — WolfGenetic")
    plt.grid(True, alpha=0.3)
    _guardar("wolf_convergencia.png")

    # 2. Contracción del radio
    plt.figure(figsize=(9, 4))
    plt.plot(iters, h["radio"], marker="s", color="steelblue", linewidth=2)
    plt.xlabel("Iteración GWO"); plt.ylabel("Radio del GA local")
    plt.title("Contracción del radio por iteración")
    plt.grid(True, alpha=0.3)
    _guardar("wolf_radio.png")

    # 3. Distribución de la población por iteración (boxplot)
    acc_lobos = [[1 - s for s in sc] for sc in h["scores_lobos"]]
    plt.figure(figsize=(12, 5))
    plt.boxplot(acc_lobos, positions=iters, widths=0.6, patch_artist=True,
                boxprops=dict(facecolor="lightsteelblue", color="navy"),
                medianprops=dict(color="red", linewidth=2))
    plt.xlabel("Iteración GWO"); plt.ylabel("Accuracy de la manada (validación)")
    plt.title("Distribución de la población por iteración")
    plt.grid(True, alpha=0.3, axis="y")
    _guardar("wolf_boxplot.png")

    # 4. Tendencia central (media y mediana junto al alfa)
    plt.figure(figsize=(9, 4))
    plt.plot(iters, acc_alpha, marker="o", color="darkblue",  linewidth=2, label="Alfa (mejor)")
    plt.plot(iters, acc_media, marker="^", color="steelblue", linewidth=2, label="Media")
    plt.plot(iters, acc_medi,  marker="s", color="tomato",    linewidth=2, label="Mediana")
    plt.xlabel("Iteración GWO"); plt.ylabel("Accuracy (validación)")
    plt.title("Tendencia central de la manada por iteración")
    plt.legend(); plt.grid(True, alpha=0.3)
    _guardar("wolf_tendencia_central.png")

    # 5. Matriz de confusión del modelo final
    _matriz_confusion(datos["matriz_confusion"], datos["clases"],
                      "Matriz de Confusión — WolfGenetic",
                      "wolf_matriz_confusion.png")


def graficas_mlp(datos):
    """Genera la matriz de confusión del MLP base."""
    _matriz_confusion(datos["matriz_confusion"], datos["clases"],
                      "Matriz de Confusión — MLP Base",
                      "mlp_matriz_confusion.png")


def main():
    print("=" * 55)
    print("  GENERACIÓN DE GRÁFICAS")
    print("=" * 55)

    # MLP puro
    jmlp = ultimo_json("mlp_puro")
    if jmlp:
        print(f"\nMLP puro  ← {os.path.basename(jmlp)}")
        with open(jmlp, encoding="utf-8") as f:
            graficas_mlp(json.load(f))
    else:
        print("\nMLP puro  → sin datos (corre primero la simulación 1)")

    # WolfGenetic
    jwolf = ultimo_json("wolfgenetic")
    if jwolf:
        print(f"\nWolfGenetic  ← {os.path.basename(jwolf)}")
        with open(jwolf, encoding="utf-8") as f:
            graficas_wolfgenetic(json.load(f))
    else:
        print("\nWolfGenetic  → sin datos (corre primero la simulación 2)")

    print(f"\nListo. Gráficas en: graficas/")


if __name__ == "__main__":
    main()
