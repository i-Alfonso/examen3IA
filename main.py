"""
main.py — Menú principal del proyecto WolfGenetic

Punto de entrada. Lee la configuración desde config.yaml y permite lanzar:
  1. La red neuronal pura (MLP base)
  2. El WolfGenetic (GWO + GA)
  3. La generación de gráficas a partir de los resultados guardados

Cada simulación corre como un subproceso para aislar el entorno (esto
también permite aplicar el fix de threads que evita el deadlock de joblib).

Ejecutar:  python3 main.py
"""
import os
import sys
import subprocess
import yaml

# ─── Colores ANSI ──────────────────────────────────────────────────
R  = "\033[0m";  B  = "\033[1m"
AZ = "\033[94m"; CY = "\033[96m"; VE = "\033[92m"
RO = "\033[91m"; AM = "\033[93m"; GR = "\033[90m"

RAIZ = os.path.dirname(os.path.abspath(__file__))

# Entorno para los subprocesos:
#  - PYTHONPATH=. para que encuentren el paquete utils/
#  - threads de BLAS = 1 para evitar el deadlock de joblib (oversubscription)
ENV = {
    **os.environ,
    "PYTHONPATH": RAIZ,
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def cargar_config():
    with open(os.path.join(RAIZ, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def correr(nombre, script):
    """Lanza un script de src/ como subproceso, mostrando su salida en vivo."""
    print(f"\n{AZ}{'─'*55}{R}")
    print(f"{B}{AZ}  ▶ {nombre}{R}")
    print(f"{AZ}{'─'*55}{R}\n")

    ruta = os.path.join(RAIZ, "src", script)
    proceso = subprocess.Popen([sys.executable, "-u", ruta], env=ENV)
    cancelado = False
    try:
        proceso.wait()
    except KeyboardInterrupt:
        proceso.terminate()
        try:
            proceso.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proceso.kill()
        cancelado = True

    if cancelado:
        print(f"\n{AM}{B}  ⚠ {nombre} cancelado.{R}")
    elif proceso.returncode != 0:
        print(f"\n{RO}{B}  ✗ Error en {nombre}.{R}")
    else:
        print(f"\n{VE}{B}  ✓ {nombre} completado.{R}")

    input(f"\n{GR}  Enter para volver al menú...{R}")


def mostrar_menu(cfg):
    os.system("clear")
    d = cfg["dataset"]
    print(f"\n{AZ}{B}  ╔{'═'*45}╗{R}")
    print(f"{AZ}{B}  ║{'WolfGenetic — Menú Principal':^45}║{R}")
    print(f"{AZ}{B}  ╚{'═'*45}╝{R}")
    print(f"\n{GR}  Dataset: {d['nombre']} | split {int(d['train']*100)}/"
          f"{int(d['val']*100)}/{int(d['test']*100)} | scaler={d['scaler']}{R}")
    print(f"{GR}  Parámetros: config.yaml{R}")
    print(f"\n{CY}  {'─'*45}{R}")
    print(f"  {B}{AM}[1]{R}  Red neuronal pura (MLP base)")
    print(f"  {B}{AM}[2]{R}  WolfGenetic (GWO + Genético)")
    print(f"  {B}{AM}[3]{R}  Generar gráficas (desde resultados)")
    print(f"  {B}{AM}[4]{R}  Correr todo (1 → 2 → 3)")
    print(f"  {B}{AM}[0]{R}  {RO}Salir{R}")
    print(f"{CY}  {'─'*45}{R}")


def menu():
    cfg = cargar_config()
    mostrar_menu(cfg)
    while True:
        op = input(f"\n{B}  ▶ Opción: {R}").strip()

        if op == "1":
            correr("Red neuronal pura", "sim_mlp_puro.py")
            mostrar_menu(cfg)
        elif op == "2":
            correr("WolfGenetic", "sim_wolfgenetic.py")
            mostrar_menu(cfg)
        elif op == "3":
            correr("Generar gráficas", "generar_graficas.py")
            mostrar_menu(cfg)
        elif op == "4":
            correr("Red neuronal pura", "sim_mlp_puro.py")
            correr("WolfGenetic", "sim_wolfgenetic.py")
            correr("Generar gráficas", "generar_graficas.py")
            mostrar_menu(cfg)
        elif op == "0":
            os.system("clear")
            print(f"\n{VE}{B}  WolfGenetic finalizado.{R}\n")
            break
        else:
            print(f"{RO}  Opción no válida.{R}")


if __name__ == "__main__":
    menu()
