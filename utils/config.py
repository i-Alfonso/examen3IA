"""
config.py — Carga centralizada del archivo config.yaml

Cualquier módulo que necesite parámetros los lee desde aquí.
Así, para cambiar la configuración del experimento, solo se edita
config.yaml (en la raíz del proyecto) y no hay que tocar el código.
"""
import os
import yaml

# Ruta al config.yaml en la raíz del proyecto (examen3/)
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def cargar_config():
    """Lee y devuelve el contenido de config.yaml como diccionario."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Se carga una sola vez al importar el módulo
CONFIG = cargar_config()


def limites_busqueda():
    """
    Devuelve (LB, UB): las listas de límites inferior y superior del
    espacio de búsqueda, en el orden que esperan el GA y el GWO:
        [lr, alpha, neuronas, capas, activacion, max_iter]
    """
    e = CONFIG["espacio_busqueda"]
    LB = [e["learning_rate_init"][0], e["alpha"][0], e["neuronas"][0],
          e["capas"][0], 0, e["max_iter"][0]]
    UB = [e["learning_rate_init"][1], e["alpha"][1], e["neuronas"][1],
          e["capas"][1], 2, e["max_iter"][1]]
    return LB, UB
