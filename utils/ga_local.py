import random
import numpy as np

# Límites del hiperespacio de búsqueda (6 dimensiones):
#   [lr,      alpha,    neuronas, capas, activacion, max_iter]
LB = [0.0001,   0.000001, 10,  1, 0, 100]
UB = [0.1,      0.01,     500, 4, 2, 500]
DIM = 6  # dimensiones del vector de hiperparámetros


def _individuo_aleatorio(lb, ub):
    # Muestreo uniforme en todo el hiperespacio (GA autónomo sin restricción de zona)
    return [random.uniform(lb[d], ub[d]) for d in range(DIM)]


def _individuo_en_vecindad(centro, radio, lb, ub):
    """Genera un individuo dentro del radio alrededor del centro.

    El radio es relativo al rango de cada dimensión, por lo que una dimensión
    de rango 490 (neuronas) y una de 0.0999 (lr) tienen la misma cobertura
    proporcional.
    """
    ind = []
    for d in range(DIM):
        rango = ub[d] - lb[d]
        valor = centro[d] + random.uniform(-radio, radio) * rango
        ind.append(max(lb[d], min(ub[d], valor)))
    return ind


def _torneo(poblacion, scores, k=2):
    """Selecciona el mejor de k individuos elegidos al azar (presión selectiva baja).

    k=2 (torneo binario) equilibra exploración y explotación; k mayor
    aumentaría la presión selectiva pero reduce diversidad.
    """
    candidatos = random.sample(range(len(poblacion)), k)
    mejor = min(candidatos, key=lambda i: scores[i])
    return poblacion[mejor][:]


def _cruce(padre1, padre2):
    """
    Cruce mixto adaptado a la naturaleza de cada dimensión:
      - Continuas (lr, alpha, max_iter): cruce aritmético (mezcla convexa).
      - Discretas  (neuronas, capas, activacion): cruce uniforme (elige uno u otro).

    El cruce aritmético en enteros generaría valores intermedios sin sentido
    para la arquitectura (¿2.7 capas?), de ahí la separación.
    """
    hijo = []
    discretas = {2, 3, 4}  # índices de dimensiones discretas
    for d in range(DIM):
        if d in discretas:
            # Hereda directamente de uno de los dos padres
            hijo.append(padre1[d] if random.random() < 0.5 else padre2[d])
        else:
            # Combinación convexa: α·p1 + (1-α)·p2
            alpha = random.random()
            hijo.append(alpha * padre1[d] + (1 - alpha) * padre2[d])
    return hijo


def _mutar(individuo, radio, lb, ub, p_mutacion=0.2):
    """Mutación gaussiana escalada por el radio actual.

    Cuando el GWO contrae el radio en iteraciones avanzadas, la desviación
    estándar de la gaussiana también se reduce → búsqueda local más fina.
    """
    for d in range(DIM):
        if random.random() < p_mutacion:
            rango = ub[d] - lb[d]
            individuo[d] += random.gauss(0, radio * rango)
            individuo[d] = max(lb[d], min(ub[d], individuo[d]))
    return individuo


def ga_local(func_objetivo, centro=None, radio=1.0,
             lb=LB, ub=UB,
             n_individuos=8, n_generaciones=3,
             p_cruce=0.8, p_mutacion=0.2):
    """
    Algoritmo Genético local.

    Si se pasa centro y radio, genera individuos en la vecindad del centro
    (usado por el GWO). Si no, genera individuos en todo el espacio [lb, ub]
    (usado en paso2_ga_solo.py de forma autónoma).

    Args:
        func_objetivo : función que recibe un individuo y devuelve el error
        centro        : posición del lobo (None = GA autónomo)
        radio         : radio de vecindad relativo al rango de cada dimensión
        lb / ub       : límites del hiperespacio
        n_individuos  : tamaño de la población
        n_generaciones: número de generaciones
        p_cruce       : probabilidad de cruce
        p_mutacion    : probabilidad de mutación por gen

    Returns:
        mejor_individuo : vector de 6 hiperparámetros
        mejor_score     : error (1 - accuracy) del mejor individuo
        historial       : lista con el mejor score por generación
    """
    # Inicializar población: en la vecindad del lobo (GWO) o en todo el espacio (GA autónomo)
    if centro is None:
        poblacion = [_individuo_aleatorio(lb, ub) for _ in range(n_individuos)]
    else:
        poblacion = [_individuo_en_vecindad(centro, radio, lb, ub)
                     for _ in range(n_individuos)]
        # Garantiza que la posición actual del lobo siempre compita;
        # sin esto podría descartarse incluso si ya es buena.
        poblacion[0] = centro[:]

    scores = [func_objetivo(ind) for ind in poblacion]

    mejor_idx = scores.index(min(scores))
    mejor_individuo = poblacion[mejor_idx][:]
    mejor_score = scores[mejor_idx]
    historial = [mejor_score]  # guarda el mejor score de cada generación

    for _ in range(n_generaciones):
        # Elitismo: el mejor individuo pasa directamente a la siguiente generación
        # sin cruce ni mutación, garantizando que el score no retrocede.
        nueva_poblacion = [mejor_individuo[:]]
        nueva_scores    = [mejor_score]

        while len(nueva_poblacion) < n_individuos:
            padre1 = _torneo(poblacion, scores)
            padre2 = _torneo(poblacion, scores)

            if random.random() < p_cruce:
                hijo = _cruce(padre1, padre2)
            else:
                hijo = padre1[:]  # reproducción sin cruce (clonación del padre1)

            hijo = _mutar(hijo, radio, lb, ub, p_mutacion)
            score_hijo = func_objetivo(hijo)

            nueva_poblacion.append(hijo)
            nueva_scores.append(score_hijo)

        poblacion = nueva_poblacion
        scores    = nueva_scores

        # Actualizar el mejor global solo si hay mejora (el elitismo ya lo preserva,
        # pero este bloque registra el historial correctamente).
        mejor_idx = scores.index(min(scores))
        if scores[mejor_idx] < mejor_score:
            mejor_score     = scores[mejor_idx]
            mejor_individuo = poblacion[mejor_idx][:]

        historial.append(mejor_score)

    return mejor_individuo, mejor_score, historial


if __name__ == "__main__":
    from utils.funcion_objetivo import evaluar

    random.seed(42)
    print("=" * 50)
    print("  GA Local — prueba autónoma (sin GWO)")
    print("=" * 50)
    print(f"\nPoblación: 15 individuos | Generaciones: 10\n")

    mejor, score, hist = ga_local(
        func_objetivo = evaluar,
        centro        = None,       # sin centro → GA autónomo
        radio         = 1.0,
        n_individuos  = 15,
        n_generaciones= 10,
    )

    print("\nEvolución del mejor score por generación:")
    for g, s in enumerate(hist):
        barra = "█" * int((1 - s) * 40)
        print(f"  Gen {g:2d} | error={s:.4f} | acc={1-s:.4f} | {barra}")

    print(f"\nMejor individuo encontrado:")
    print(f"  lr        : {mejor[0]:.6f}")
    print(f"  alpha     : {mejor[1]:.8f}")
    print(f"  neuronas  : {int(round(mejor[2]))}")
    print(f"  capas     : {int(round(mejor[3]))}")
    print(f"  activacion: {['relu','tanh','logistic'][int(round(mejor[4]))%3]}")
    print(f"  max_iter  : {int(round(mejor[5]))}")
    print(f"\nAccuracy final: {(1-score)*100:.2f}%")
