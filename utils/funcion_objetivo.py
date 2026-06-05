"""
funcion_objetivo.py — El puente entre la optimización y la red neuronal

Esta es la pieza central del proyecto. El GA y el GWO nunca tocan el MLP
directamente: solo llaman a evaluar(), que recibe un vector de 6
hiperparámetros, entrena una red y devuelve qué tan mal clasifica.

Los topes de cada hiperparámetro se leen desde config.yaml.
"""
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

from utils.config import CONFIG
from utils.preprocesamiento import cargar_datos

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Los datos se cargan una sola vez al importar el módulo
X_train, X_val, X_test, y_train, y_val, y_test = cargar_datos()

# Mapeo del gen de activación (entero) al nombre que entiende sklearn
_ACTIVACIONES = {0: "relu", 1: "tanh", 2: "logistic"}

# Topes del espacio de búsqueda (desde config.yaml)
_E = CONFIG["espacio_busqueda"]
_N_MIN, _N_MAX = _E["neuronas"]
_C_MIN, _C_MAX = _E["capas"]
_I_MIN, _I_MAX = _E["max_iter"]


def construir_mlp(individuo):
    """
    Traduce un vector de 6 valores a un MLPClassifier de sklearn.

    Individuo = [lr, alpha, neuronas, capas, activacion, max_iter]

    Cada valor se acota a su rango válido (los topes vienen de config.yaml).
    Esto garantiza que aunque el GA/GWO generen un valor fuera de rango,
    la red siempre se construye con parámetros legales.
    """
    lr         = float(individuo[0])
    alpha      = float(individuo[1])
    neuronas   = max(_N_MIN, min(_N_MAX, int(round(individuo[2]))))
    capas      = max(_C_MIN, min(_C_MAX, int(round(individuo[3]))))
    activacion = _ACTIVACIONES[int(round(individuo[4])) % 3]
    max_iter   = max(_I_MIN, min(_I_MAX, int(round(individuo[5]))))

    return MLPClassifier(
        hidden_layer_sizes  = tuple([neuronas] * capas),
        learning_rate_init  = lr,
        alpha               = alpha,
        activation          = activacion,
        max_iter            = max_iter,
        early_stopping      = True,    # corta el entrenamiento si no mejora
        validation_fraction = 0.1,
        n_iter_no_change    = 10,
        random_state        = CONFIG["dataset"]["random_state"],
    )


def evaluar(individuo):
    """
    Entrena un MLP con los hiperparámetros del individuo y devuelve el
    error en validación (1 - accuracy). El GA y el GWO MINIMIZAN esto.

    Returns:
        error : float entre 0.0 (perfecto) y 1.0 (pésimo)
    """
    modelo = construir_mlp(individuo)
    modelo.fit(X_train, y_train)
    accuracy = accuracy_score(y_val, modelo.predict(X_val))
    return 1.0 - accuracy


def decodificar(individuo):
    """Devuelve los hiperparámetros legibles de un individuo (para reportes)."""
    return {
        "learning_rate_init": float(individuo[0]),
        "alpha":              float(individuo[1]),
        "neuronas":           max(_N_MIN, min(_N_MAX, int(round(individuo[2])))),
        "capas":              max(_C_MIN, min(_C_MAX, int(round(individuo[3])))),
        "activacion":         _ACTIVACIONES[int(round(individuo[4])) % 3],
        "max_iter":           max(_I_MIN, min(_I_MAX, int(round(individuo[5])))),
    }


if __name__ == "__main__":
    import time
    individuo_prueba = [0.001, 0.0001, 100, 2, 0, 300]
    print("Probando evaluar() con un individuo fijo:")
    print(f"  Hiperparámetros: {decodificar(individuo_prueba)}")
    t0 = time.time()
    error = evaluar(individuo_prueba)
    print(f"  Accuracy validación: {(1-error)*100:.2f}%  ({time.time()-t0:.2f}s)")
