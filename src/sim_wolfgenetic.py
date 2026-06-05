"""
sim_wolfgenetic.py — Simulación 2: WolfGenetic (GWO + GA local)

Ajusta los hiperparámetros del MLP usando el híbrido Grey Wolf Optimizer +
Algoritmo Genético. Toda la configuración sale de config.yaml.

Guarda log + datos (incluido el historial de iteraciones, necesario para
las gráficas) en resultados/wolfgenetic/.

Se ejecuta desde main.py o directamente:
    PYTHONPATH=. python3 src/sim_wolfgenetic.py
"""
import time
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix)

from utils.config import CONFIG, limites_busqueda
from utils.preprocesamiento import cargar_datos, clases_nombres
from utils.funcion_objetivo import evaluar, construir_mlp, decodificar
from utils.gwo import gwo
from utils.registro import Registro


def main():
    reg  = Registro("wolfgenetic")
    cfg  = CONFIG["wolfgenetic"]
    dcfg = CONFIG["dataset"]
    LB, UB = limites_busqueda()

    # ── Cabecera del log: configuración inicial ──
    print("=" * 60)
    print("  SIMULACIÓN 2 — WOLFGENETIC (GWO + GA local)")
    print(f"  Fecha: {reg.ts}")
    print("=" * 60)
    print("\nCONFIGURACIÓN INICIAL")
    print(f"  Dataset : {dcfg['nombre']}  (split {int(dcfg['train']*100)}/"
          f"{int(dcfg['val']*100)}/{int(dcfg['test']*100)}, scaler={dcfg['scaler']})")
    print(f"  Semilla : {dcfg['random_state']}")
    print("\nESPACIO DE BÚSQUEDA")
    print(f"  learning_rate_init : [{LB[0]}, {UB[0]}]")
    print(f"  alpha              : [{LB[1]}, {UB[1]}]")
    print(f"  neuronas           : [{LB[2]}, {UB[2]}]")
    print(f"  capas              : [{LB[3]}, {UB[3]}]")
    print(f"  max_iter (red)     : [{LB[5]}, {UB[5]}]")
    print("\nPARÁMETROS DEL ALGORITMO")
    print(f"  Representación GA  → {cfg.get('tipo_ga', 'real')}")
    print(f"  GWO  → lobos={cfg['n_lobos']}, iteraciones={cfg['max_iter']}, "
          f"r_max={cfg['r_max']}")
    print(f"  GA   → individuos={cfg['n_local']}, generaciones={cfg['gen_locales']}, "
          f"p_cruce={cfg['p_cruce']}, p_mutacion={cfg['p_mutacion']}")
    print(f"  Early stopping GA  → paciencia={cfg['paciencia']}")
    mlps = cfg['n_lobos'] * cfg['max_iter'] * cfg['n_local'] * cfg['gen_locales']
    print(f"  MLPs máximos       → {mlps}  (menos con early stopping)")

    # ── Datos ──
    X_tr, X_v, X_ts, y_tr, y_v, y_ts = cargar_datos()
    print(f"\n  Train={len(X_tr)} | Val={len(X_v)} | Test={len(X_ts)} | "
          f"Clases={len(clases_nombres())}")

    # ── Ejecutar WolfGenetic ──
    print(f"\n[{reg.stamp()}] Iniciando optimización...")
    t0 = time.time()
    alpha_pos, alpha_score, historial = gwo(
        evaluar,
        n_lobos      = cfg["n_lobos"],
        max_iter     = cfg["max_iter"],
        r_max        = cfg["r_max"],
        n_local      = cfg["n_local"],
        gen_locales  = cfg["gen_locales"],
        p_cruce      = cfg["p_cruce"],
        p_mutacion   = cfg["p_mutacion"],
        lb           = LB,
        ub           = UB,
        paciencia_ga = cfg["paciencia"],
        verbose      = True,
    )
    dt = time.time() - t0

    # ── Entrenar el MLP final con los mejores hiperparámetros ──
    mejores = decodificar(alpha_pos)
    modelo = construir_mlp(alpha_pos)
    modelo.fit(X_tr, y_tr)
    pred = modelo.predict(X_ts)
    acc  = accuracy_score(y_ts, pred)
    bacc = balanced_accuracy_score(y_ts, pred)

    print(f"\n[{reg.stamp()}] Optimización terminada ({dt/60:.1f} min)")
    print("\nMEJORES HIPERPARÁMETROS ENCONTRADOS")
    for k, v in mejores.items():
        print(f"  {k:<20}: {v}")
    print("\nRESULTADOS (sobre test)")
    print(f"  Accuracy          : {acc*100:.2f}%")
    print(f"  Balanced accuracy : {bacc*100:.2f}%")

    # ── Guardar datos para las gráficas ──
    #    El historial trae, por iteración: alpha_score, media, mediana,
    #    scores_lobos y radio. Es lo que necesita generar_graficas.py.
    reg.guardar_datos({
        "tipo": "wolfgenetic",
        "timestamp": reg.ts,
        "config": cfg,
        "mejores_hiperparametros": mejores,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "tiempo_seg": dt,
        "historial": historial,
        "matriz_confusion": confusion_matrix(y_ts, pred).tolist(),
        "clases": clases_nombres(),
    })

    print(f"\n[{reg.stamp()}] Simulación WolfGenetic completada.")
    reg.cerrar()
    return acc, bacc


if __name__ == "__main__":
    main()
