"""
sim_mlp_puro.py — Simulación 1: Red neuronal pura (MLP base)

Entrena un MLP con hiperparámetros FIJOS (los de config.yaml), sin ningún
tipo de optimización. Es la línea base contra la que se compara el
WolfGenetic.

Guarda log + datos en resultados/mlp_puro/.
Se ejecuta desde main.py o directamente:
    PYTHONPATH=. python3 src/sim_mlp_puro.py
"""
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix)

from utils.config import CONFIG
from utils.preprocesamiento import cargar_datos, clases_nombres
from utils.registro import Registro


def main():
    reg = Registro("mlp_puro")
    cfg = CONFIG["mlp_puro"]
    dcfg = CONFIG["dataset"]

    # ── Cabecera del log: configuración inicial ──
    print("=" * 60)
    print("  SIMULACIÓN 1 — RED NEURONAL PURA (MLP base)")
    print(f"  Fecha: {reg.ts}")
    print("=" * 60)
    print("\nCONFIGURACIÓN INICIAL")
    print(f"  Dataset : {dcfg['nombre']}  (split {int(dcfg['train']*100)}/"
          f"{int(dcfg['val']*100)}/{int(dcfg['test']*100)}, scaler={dcfg['scaler']})")
    print(f"  Semilla : {dcfg['random_state']}")
    print("\nHIPERPARÁMETROS (fijos, sin optimizar)")
    print(f"  hidden_layer_sizes : {tuple(cfg['hidden_layer_sizes'])}")
    print(f"  activation         : {cfg['activation']}")
    print(f"  learning_rate_init : {cfg['learning_rate_init']}")
    print(f"  alpha              : {cfg['alpha']}")
    print(f"  max_iter           : {cfg['max_iter']}")

    # ── Datos ──
    X_tr, X_v, X_ts, y_tr, y_v, y_ts = cargar_datos()
    print(f"\n  Train={len(X_tr)} | Val={len(X_v)} | Test={len(X_ts)} | "
          f"Clases={len(clases_nombres())}")

    # ── Entrenar ──
    print(f"\n[{reg.stamp()}] Entrenando MLP...")
    t0 = time.time()
    modelo = MLPClassifier(
        hidden_layer_sizes  = tuple(cfg["hidden_layer_sizes"]),
        activation          = cfg["activation"],
        learning_rate_init  = cfg["learning_rate_init"],
        alpha               = cfg["alpha"],
        max_iter            = cfg["max_iter"],
        early_stopping      = True,
        validation_fraction = 0.1,
        n_iter_no_change    = 10,
        random_state        = dcfg["random_state"],
    )
    modelo.fit(X_tr, y_tr)
    dt = time.time() - t0

    # ── Evaluar ──
    pred = modelo.predict(X_ts)
    acc  = accuracy_score(y_ts, pred)
    bacc = balanced_accuracy_score(y_ts, pred)

    print(f"[{reg.stamp()}] Entrenamiento terminado ({dt:.1f}s, "
          f"{modelo.n_iter_} épocas)")
    print("\nRESULTADOS (sobre test)")
    print(f"  Accuracy          : {acc*100:.2f}%")
    print(f"  Balanced accuracy : {bacc*100:.2f}%")

    # ── Guardar datos para las gráficas ──
    reg.guardar_datos({
        "tipo": "mlp_puro",
        "timestamp": reg.ts,
        "config": cfg,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "tiempo_seg": dt,
        "matriz_confusion": confusion_matrix(y_ts, pred).tolist(),
        "clases": clases_nombres(),
    })

    print(f"\n[{reg.stamp()}] Simulación MLP puro completada.")
    reg.cerrar()
    return acc, bacc


if __name__ == "__main__":
    main()
