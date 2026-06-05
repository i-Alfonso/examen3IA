import time
import warnings
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from utils.preprocesamiento import cargar_datos

# Carga única al importar el módulo: evita releer disco en cada llamada a evaluar(),
# que ocurre cientos de veces durante la optimización (GA/GWO).
X_train, X_val, X_test, y_train, y_val, y_test = cargar_datos()

# El GA/GWO codifican la activación como entero continuo [0, 2].
# El % 3 garantiza que valores fuera del rango sigan siendo válidos.
_ACTIVACIONES = {0: "relu", 1: "tanh", 2: "logistic"}


def evaluar(individuo):
    """
    Entrena un MLP con los hiperparámetros del individuo y devuelve el error
    en validación (1 - accuracy). El GA y el GWO minimizan este valor.

    Individuo — vector de 6 valores continuos:
        [0] learning_rate_init  → float  en [0.0001, 0.1]
        [1] alpha (reg. L2)     → float  en [0.000001, 0.01]
        [2] neuronas por capa   → int    en [10, 500]
        [3] número de capas     → int    en [1, 4]
        [4] activación          → int    en {0=relu, 1=tanh, 2=logistic}
        [5] max_iter            → int    en [100, 500]

    Returns:
        error : float  (0.0 = perfecto, 1.0 = pésimo)
    """
    # Decodificación: GA/GWO trabajan con floats continuos; el MLP necesita
    # enteros para neuronas/capas y string para la función de activación.
    lr        = float(individuo[0])
    alpha     = float(individuo[1])
    neuronas  = max(10,  int(round(individuo[2])))          # mínimo 10 neuronas por capa
    capas     = max(1,   min(4, int(round(individuo[3]))))  # entre 1 y 4 capas ocultas
    activacion = _ACTIVACIONES[int(round(individuo[4])) % 3]
    max_iter  = max(100, min(500, int(round(individuo[5]))))

    # Arquitectura uniforme: todas las capas tienen el mismo nº de neuronas.
    # Ej: neuronas=100, capas=2  →  hidden_layer_sizes=(100, 100)
    modelo = MLPClassifier(
        hidden_layer_sizes = tuple([neuronas] * capas),
        learning_rate_init = lr,
        alpha              = alpha,
        activation         = activacion,
        max_iter           = max_iter,
        # Early stopping interno del MLP (sobre fracción del train),
        # independiente de la validación externa usada para calcular el fitness.
        early_stopping     = True,
        validation_fraction= 0.1,
        n_iter_no_change   = 10,
        random_state       = 42
    )

    modelo.fit(X_train, y_train)
    # El fitness se evalúa sobre X_val para que X_test quede completamente
    # reservado al reporte final y no influya en la búsqueda.
    accuracy = accuracy_score(y_val, modelo.predict(X_val))
    return 1.0 - accuracy  # minimizar error ≡ maximizar accuracy


if __name__ == "__main__":
    # Individuo de prueba con hiperparámetros razonables
    individuo_prueba = [
        0.001,   # learning_rate_init
        0.0001,  # alpha
        100,     # neuronas por capa
        2,       # número de capas → (100, 100)
        0,       # activación → relu
        300,     # max_iter
    ]

    print("Probando funcion_objetivo con individuo fijo:")
    print(f"  lr={individuo_prueba[0]}, alpha={individuo_prueba[1]}, "
          f"neuronas={individuo_prueba[2]}, capas={individuo_prueba[3]}, "
          f"activacion={_ACTIVACIONES[individuo_prueba[4]]}, max_iter={individuo_prueba[5]}")
    print(f"  Arquitectura MLP: {tuple([individuo_prueba[2]] * individuo_prueba[3])}")
    print()

    t0    = time.time()
    error = evaluar(individuo_prueba)
    dt    = time.time() - t0

    print(f"  Error     : {error:.4f}")
    print(f"  Accuracy  : {1 - error:.4f}  ({(1-error)*100:.2f}%)")
    print(f"  Tiempo    : {dt:.2f}s")
    print()
    print("Criterio de aceptacion: accuracy >= 0.93")
    print(f"  {'PASA' if (1 - error) >= 0.93 else 'FALLA'}")
