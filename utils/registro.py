"""
registro.py — Guardado ordenado de logs y datos de cada simulación

Cada simulación (MLP puro o WolfGenetic) genera:
  - un log de texto legible  → resultados/<tipo>/log_<TS>.txt
  - un JSON con los datos     → resultados/<tipo>/datos_<TS>.json

El log es para leerlo a mano (incluye parámetros, configuración y resultados).
El JSON es para que generar_graficas.py lo lea después y dibuje, sin tener
que volver a entrenar nada.
"""
import os
import sys
import json
import datetime

# Carpeta raíz de resultados (dentro de examen3/)
_RES = os.path.join(os.path.dirname(__file__), "..", "resultados")


class Registro:
    """Maneja el log de texto y el guardado de datos de una simulación."""

    def __init__(self, tipo):
        """tipo: 'mlp_puro' o 'wolfgenetic' (define la subcarpeta)."""
        self.tipo = tipo
        self.ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir  = os.path.join(_RES, tipo)
        os.makedirs(self.dir, exist_ok=True)

        self.log_path  = os.path.join(self.dir, f"log_{self.ts}.txt")
        self.json_path = os.path.join(self.dir, f"datos_{self.ts}.json")

        # Redirige stdout para que todo print() vaya también al log
        self._archivo = open(self.log_path, "w", encoding="utf-8")
        self._stdout  = sys.__stdout__
        sys.stdout = self

    # ── Tee: escribe en consola y en el archivo a la vez ──
    def write(self, msg):
        self._stdout.write(msg)
        self._archivo.write(msg)
        self._archivo.flush()

    def flush(self):
        self._stdout.flush()
        self._archivo.flush()

    # ── Utilidades ──
    def stamp(self):
        return datetime.datetime.now().strftime("%H:%M:%S")

    def guardar_datos(self, datos):
        """Guarda el diccionario de datos como JSON (para las gráficas)."""
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
        print(f"\n[{self.stamp()}] Datos guardados → {self.json_path}")

    def cerrar(self):
        sys.stdout = self._stdout
        self._archivo.close()


def ultimo_json(tipo):
    """Devuelve la ruta del JSON más reciente de un tipo de simulación."""
    d = os.path.join(_RES, tipo)
    if not os.path.isdir(d):
        return None
    jsons = [f for f in os.listdir(d) if f.startswith("datos_") and f.endswith(".json")]
    if not jsons:
        return None
    return os.path.join(d, sorted(jsons)[-1])
