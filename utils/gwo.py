import random
from joblib import Parallel, delayed

from utils.ga_local import ga_local
from utils.ga_local_bin import ga_local_bin
from utils.config import limites_busqueda, CONFIG

# Límites del hiperespacio leídos desde config.yaml
LB, UB = limites_busqueda()
DIM = 6

# El tipo de GA local se elige desde config.yaml ("real" o "binario").
# Ambos tienen la misma interfaz, así que son intercambiables.
_TIPO_GA = CONFIG["wolfgenetic"].get("tipo_ga", "real")
_GA = ga_local_bin if _TIPO_GA == "binario" else ga_local


def _lobo_aleatorio(lb, ub):
    return [random.uniform(lb[d], ub[d]) for d in range(DIM)]


def _mover_lobo(lobo, alpha, beta, delta, a, lb, ub):
    """Fórmula estándar del GWO: promedio de los 3 jalones de los líderes."""
    nueva_pos = []
    for d in range(DIM):
        jalon = 0.0
        for lider in [alpha, beta, delta]:
            r1 = random.random()
            r2 = random.random()
            A  = 2 * a * r1 - a
            C  = 2 * r2
            D  = abs(C * lider[d] - lobo[d])
            jalon += lider[d] - A * D
        nueva_pos.append(max(lb[d], min(ub[d], jalon / 3)))
    return nueva_pos


def _actualizar_jerarquia(alpha, alpha_score, beta, beta_score,
                          delta, delta_score, pos, score):
    """Actualiza α, β, δ si el nuevo lobo es mejor que alguno de ellos."""
    if score < alpha_score:
        delta, delta_score = beta[:], beta_score
        beta,  beta_score  = alpha[:], alpha_score
        alpha, alpha_score = pos[:], score
    elif score < beta_score:
        delta, delta_score = beta[:], beta_score
        beta,  beta_score  = pos[:], score
    elif score < delta_score:
        delta, delta_score = pos[:], score
    return alpha, alpha_score, beta, beta_score, delta, delta_score


def gwo(func_objetivo, n_lobos=10, max_iter=30, r_max=0.3,
        n_local=8, gen_locales=3, p_cruce=0.8, p_mutacion=0.2,
        lb=LB, ub=UB, verbose=True, paciencia_ga=None):
    """
    Grey Wolf Optimizer con GA local por lobo (WolfGenetic).

    Cada iteración:
      1. Cada lobo lanza su GA local en su vecindad (radio decrece con 'a')
      2. La mejor posición del GA actualiza al lobo
      3. Se actualiza la jerarquía α, β, δ
      4. GWO mueve a cada lobo hacia los líderes

    Args:
        func_objetivo : función que recibe un individuo y devuelve error (float)
        n_lobos       : tamaño de la manada
        max_iter      : iteraciones del GWO
        r_max         : radio máximo del GA local (fracción del rango)
        n_local       : individuos por GA local
        gen_locales   : generaciones por GA local
        p_cruce       : probabilidad de cruce en el GA local
        p_mutacion    : probabilidad de mutación en el GA local
        lb / ub       : límites del hiperespacio
        verbose       : imprimir progreso por iteración

    Returns:
        alpha_pos     : mejores hiperparámetros encontrados
        alpha_score   : error del alpha (1 - accuracy)
        historial     : dict con listas de métricas por iteración
    """
    # ── Inicializar manada ────────────────────────────────────────
    manada = [_lobo_aleatorio(lb, ub) for _ in range(n_lobos)]
    scores = [func_objetivo(lobo) for lobo in manada]

    alpha, alpha_score = manada[0][:], scores[0]
    beta,  beta_score  = manada[0][:], scores[0]
    delta, delta_score = manada[0][:], scores[0]

    for i in range(n_lobos):
        alpha, alpha_score, beta, beta_score, delta, delta_score = \
            _actualizar_jerarquia(alpha, alpha_score, beta, beta_score,
                                  delta, delta_score, manada[i], scores[i])

    historial = {
        "alpha_score" : [],   # mejor score por iteración
        "scores_lobos": [],   # todos los scores por iteración (para boxplot)
        "media"       : [],   # media de scores por iteración
        "mediana"     : [],   # mediana de scores por iteración
        "radio"       : [],   # radio del GA local por iteración
    }

    # ── Bucle principal GWO ───────────────────────────────────────
    for t in range(max_iter):
        a     = 2.0 - t * (2.0 / max_iter)
        radio = r_max * (a / 2.0)

        # Evaluar todos los lobos en paralelo (un core por lobo).
        # _GA es ga_local (real) o ga_local_bin (binario) según config.yaml.
        resultados = Parallel(n_jobs=-1)(
            delayed(_GA)(
                func_objetivo  = func_objetivo,
                centro         = manada[i],
                radio          = radio,
                lb             = lb,
                ub             = ub,
                n_individuos   = n_local,
                n_generaciones = gen_locales,
                p_cruce        = p_cruce,
                p_mutacion     = p_mutacion,
                paciencia      = paciencia_ga,
            )
            for i in range(n_lobos)
        )

        scores_iter = []
        for i, (mejor_local, score_local, _) in enumerate(resultados):
            manada[i] = mejor_local
            scores[i] = score_local
            scores_iter.append(score_local)

            alpha, alpha_score, beta, beta_score, delta, delta_score = \
                _actualizar_jerarquia(alpha, alpha_score, beta, beta_score,
                                      delta, delta_score, manada[i], scores[i])

        # GWO mueve a cada lobo hacia los líderes
        for i in range(n_lobos):
            manada[i] = _mover_lobo(manada[i], alpha, beta, delta, a, lb, ub)

        # Registrar métricas
        historial["alpha_score"].append(alpha_score)
        historial["scores_lobos"].append(scores_iter[:])
        historial["media"].append(sum(scores_iter) / len(scores_iter))
        historial["mediana"].append(sorted(scores_iter)[len(scores_iter) // 2])
        historial["radio"].append(radio)

        if verbose:
            print(f"Iter {t+1:3d}/{max_iter} | "
                  f"a={a:.3f} | radio={radio:.4f} | "
                  f"alpha={1-alpha_score:.4f} | "
                  f"media={1-historial['media'][-1]:.4f}")

    return alpha, alpha_score, historial


if __name__ == "__main__":
    from utils.funcion_objetivo import evaluar

    random.seed(42)
    print("=" * 60)
    print("  GWO — prueba rápida (config reducida)")
    print("=" * 60)
    print()

    mejor, score, hist = gwo(
        func_objetivo = evaluar,
        n_lobos       = 5,
        max_iter      = 5,
        r_max         = 0.3,
        n_local       = 4,
        gen_locales   = 2,
        verbose       = True,
    )

    print(f"\nAlpha final:")
    act = {0: "relu", 1: "tanh", 2: "logistic"}
    print(f"  lr        : {mejor[0]:.6f}")
    print(f"  alpha     : {mejor[1]:.8f}")
    print(f"  neuronas  : {int(round(mejor[2]))}")
    print(f"  capas     : {int(round(mejor[3]))}")
    print(f"  activacion: {act[int(round(mejor[4]))%3]}")
    print(f"  max_iter  : {int(round(mejor[5]))}")
    print(f"\nAccuracy alpha: {(1-score)*100:.2f}%")
    print(f"\nRadio por iteración: {[round(r,4) for r in hist['radio']]}")
    print("(debe decrecer de r_max hacia 0)")
