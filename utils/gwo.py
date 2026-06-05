import random
import math
from joblib import Parallel, delayed

from utils.ga_local import ga_local

LB = [0.0001,   0.000001, 10,  1, 0, 100]
UB = [0.1,      0.01,     500, 4, 2, 500]
DIM = 6


def _lobo_aleatorio(lb, ub):
    # Inicialización aleatoria uniforme: todos los lobos arrancan sin sesgo
    return [random.uniform(lb[d], ub[d]) for d in range(DIM)]


def _mover_lobo(lobo, alpha, beta, delta, a, lb, ub):
    """Ecuaciones de movimiento del GWO original (Mirjalili et al., 2014).

    Para cada dimensión d:
        A = 2·a·r1 - a          (coeficiente de movimiento, decrece con 'a')
        C = 2·r2                (factor de peso aleatorio)
        D = |C·lider[d] - lobo[d]|  (distancia escalada al líder)
        jalón = lider[d] - A·D       (punto destino respecto a ese líder)

    La nueva posición es el promedio de los 3 jalones (α, β, δ),
    lo que combina la influencia de los tres mejores lobos.
    """
    nueva_pos = []
    for d in range(DIM):
        jalon = 0.0
        for lider in [alpha, beta, delta]:
            r1 = random.random()
            r2 = random.random()
            A  = 2 * a * r1 - a   # cuando a→0, A∈[-1,1]: explotación local
            C  = 2 * r2
            D  = abs(C * lider[d] - lobo[d])
            jalon += lider[d] - A * D
        nueva_pos.append(max(lb[d], min(ub[d], jalon / 3)))
    return nueva_pos


def _actualizar_jerarquia(alpha, alpha_score, beta, beta_score,
                          delta, delta_score, pos, score):
    """Mantiene los 3 mejores lobos (α < β < δ) en orden estricto.

    La jerarquía se actualiza desplazando en cascada:
    si el nuevo supera al α, el anterior α pasa a β y el β a δ.
    """
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
        lb=LB, ub=UB, verbose=True):
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
    # Evaluación inicial: cada lobo es un vector de hiperparámetros candidato
    scores = [func_objetivo(lobo) for lobo in manada]

    # Inicializar jerarquía con el primer lobo para que los tres líderes
    # apunten a posiciones válidas antes del primer ciclo de actualización
    alpha, alpha_score = manada[0][:], scores[0]
    beta,  beta_score  = manada[0][:], scores[0]
    delta, delta_score = manada[0][:], scores[0]

    # Construir la jerarquía real con toda la manada inicial
    for i in range(n_lobos):
        alpha, alpha_score, beta, beta_score, delta, delta_score = \
            _actualizar_jerarquia(alpha, alpha_score, beta, beta_score,
                                  delta, delta_score, manada[i], scores[i])

    historial = {
        "alpha_score" : [],   # mejor score (alpha) por iteración
        "scores_lobos": [],   # score de cada lobo por iteración (para boxplot)
        "media"       : [],   # media de scores por iteración
        "mediana"     : [],   # mediana de scores por iteración
        "radio"       : [],   # radio del GA local (decrece con 'a')
    }

    # ── Bucle principal GWO ───────────────────────────────────────
    for t in range(max_iter):
        # 'a' decrece linealmente de 2 a 0: controla balance exploración/explotación.
        # Al inicio (a≈2) los lobos se mueven ampliamente; al final (a≈0) convergen.
        a     = 2.0 - t * (2.0 / max_iter)
        radio = r_max * (a / 2.0)  # el radio del GA local también se contrae

        # Cada lobo ejecuta su GA local en paralelo (n_jobs=-1 usa todos los cores).
        # El GA refina la posición del lobo dentro de su vecindad antes de que el
        # GWO lo mueva hacia los líderes — esto es la innovación "WolfGenetic".
        resultados = Parallel(n_jobs=-1)(
            delayed(ga_local)(
                func_objetivo  = func_objetivo,
                centro         = manada[i],
                radio          = radio,
                lb             = lb,
                ub             = ub,
                n_individuos   = n_local,
                n_generaciones = gen_locales,
                p_cruce        = p_cruce,
                p_mutacion     = p_mutacion,
            )
            for i in range(n_lobos)
        )

        scores_iter = []
        for i, (mejor_local, score_local, _) in enumerate(resultados):
            # Actualizar la posición del lobo con el mejor encontrado por su GA
            manada[i] = mejor_local
            scores[i] = score_local
            scores_iter.append(score_local)

            # Actualizar jerarquía tras la mejora local de cada lobo
            alpha, alpha_score, beta, beta_score, delta, delta_score = \
                _actualizar_jerarquia(alpha, alpha_score, beta, beta_score,
                                      delta, delta_score, manada[i], scores[i])

        # GWO clásico: mover cada lobo hacia el centroide de los 3 líderes
        for i in range(n_lobos):
            manada[i] = _mover_lobo(manada[i], alpha, beta, delta, a, lb, ub)

        # Registrar métricas de esta iteración para las gráficas del paso 3
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
