import subprocess
import sys
import os
import threading
import time
import signal

# ─── Colores ANSI ────────────────────────────────────────────────
# Constantes de escape para colorear la salida en terminal.
# Se usan en f-strings como: f"{AZ}{B}texto{R}" (azul+negrita, luego reset)
R  = "\033[0m"   # reset — vuelve al color por defecto
B  = "\033[1m"   # negrita
AZ = "\033[94m"  # azul claro
CY = "\033[96m"  # cian
VE = "\033[92m"  # verde
RO = "\033[91m"  # rojo
AM = "\033[93m"  # amarillo
GR = "\033[90m"  # gris

# Se inyecta PYTHONPATH="." para que los scripts en src/ puedan importar utils/
ENV = {**os.environ, "PYTHONPATH": "."}

# ─── Parámetros por defecto ───────────────────────────────────────
DEFAULTS = {
    "paso1": {
        "hidden_layers" : 100,
        "n_capas"       : 2,
        "activation"    : "relu",
        "lr"            : 0.001,
        "alpha"         : 0.0001,
        "max_iter"      : 300,
    },
    "paso2": {
        "n_individuos"  : 15,
        "n_generaciones": 20,
        "p_cruce"       : 0.8,
        "p_mutacion"    : 0.2,
    },
    "paso3": {
        "── GWO (Lobos Grises) ──": "─────────",
        "n_lobos"     : 10,
        "max_iter"    : 30,
        "r_max"       : 0.3,
        "── GA (Genético local) ──": "─────────",
        "n_individuos": 8,
        "n_generaciones": 3,
        "p_cruce"     : 0.8,
        "p_mutacion"  : 0.2,
    },
}


def pedir_params(paso):
    """Muestra los parámetros actuales y permite modificarlos."""
    params = dict(DEFAULTS[paso])
    if not params:
        return params

    print(f"\n{CY}  {'─'*43}{R}")
    print(f"  {B}Parámetros actuales:{R}")
    print(f"{CY}  ┌{'─'*41}┐{R}")
    for k, v in params.items():
        if k.startswith("──"):
            print(f"{CY}  │{R}  {GR}{k}{R}")
        else:
            print(f"{CY}  │{R}  {AM}{k:<16}{R}: {B}{v}{R}")
    print(f"{CY}  └{'─'*41}┘{R}")

    print(f"\n  {B}{AM}[1]{R}  Usar valores por defecto")
    print(f"  {B}{AM}[2]{R}  Modificar parámetros")

    opcion = input(f"\n{B}  ▶ Elige: {R}").strip()

    if opcion == "2":
        print(f"\n{GR}  (Enter = mantener valor actual){R}\n")
        for k, v in params.items():
            if k.startswith("──"):   # es separador, no es editable
                continue
            nuevo = input(f"  {AM}{k}{R} [{B}{v}{R}]: ").strip()
            if nuevo:
                try:
                    params[k] = type(v)(nuevo)
                except ValueError:
                    print(f"  {RO}Valor inválido, se mantiene {v}{R}")

        print(f"\n{CY}  {'─'*43}{R}")
        print(f"  {B}Parámetros finales:{R}")
        for k, v in params.items():
            print(f"  {AM}{k:<16}{R}: {VE}{v}{R}")
        print(f"{CY}  {'─'*43}{R}")

    return params


def _spinner(texto, stop_event):
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop_event.is_set():
        print(f"\r  {AM}{frames[i % len(frames)]}{R} {texto}...  "
              f"{GR}(Ctrl+C para cancelar){R}", end="", flush=True)
        time.sleep(0.1)
        i += 1
    print(f"\r{' ' * 70}\r", end="", flush=True)


def correr(nombre, archivo, params=None, prod=False):
    """
    prod=False → spinner al inicio + pide Enter al final (opciones 1,2,3)
    prod=True  → output inmediato sin spinner (opción 4)
    """
    print(f"\n{AZ}{'─'*50}{R}")
    print(f"{B}{AZ}  ▶ EJECUTANDO: {nombre}{R}")
    print(f"{AZ}{'─'*50}{R}\n")

    env = dict(ENV)
    if params:
        # Los parámetros se pasan como variables de entorno PARAM_<NOMBRE>
        # porque cada paso corre en un subproceso separado y no comparte estado.
        for k, v in params.items():
            env[f"PARAM_{k.upper()}"] = str(v)

    cancelado = False

    if prod:
        # Modo producción (opción 4): output en tiempo real sin spinner,
        # útil para pipelines automáticos donde se quiere ver el progreso al vuelo.
        proceso = subprocess.Popen([sys.executable, "-u", archivo], env=env)
        try:
            proceso.wait()
        except KeyboardInterrupt:
            proceso.terminate()
            try:
                proceso.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proceso.kill()
            cancelado = True
    else:
        # Modo interactivo: muestra spinner mientras el subproceso arranca,
        # luego lo reemplaza con el output real en cuanto llega la primera línea.
        stop = threading.Event()
        hilo = threading.Thread(target=_spinner, args=(nombre, stop), daemon=True)
        hilo.start()

        proceso = subprocess.Popen(
            [sys.executable, "-u", archivo],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # stderr redirigido a stdout para capturar todo
            text=True
        )
        primera = True
        inicio = time.time()
        try:
            for linea in proceso.stdout:
                if primera:
                    # Esperar al menos 1s para que el spinner se vea; sin esto
                    # desaparece instantáneamente si el script arranca muy rápido.
                    tiempo_minimo = 1.0
                    restante = tiempo_minimo - (time.time() - inicio)
                    if restante > 0:
                        time.sleep(restante)
                    stop.set()
                    hilo.join()
                    print()
                    primera = False
                print(linea, end="", flush=True)
            proceso.wait()
        except KeyboardInterrupt:
            proceso.terminate()
            try:
                proceso.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proceso.kill()
            cancelado = True

        stop.set()
        hilo.join()

    if cancelado:
        print(f"\n{AM}{B}  ⚠ {nombre} cancelado.{R}")
    elif proceso.returncode != 0:
        print(f"\n{RO}{B}  ✗ ERROR en {nombre}.{R}")
    else:
        print(f"\n{VE}{B}  ✓ {nombre} completado.{R}")

    if not prod:
        input(f"\n{GR}  Presiona Enter para volver al menú...{R}")

    return not cancelado and proceso.returncode == 0


def menu():
    mostrar_menu()
    while True:
        opcion = input(f"\n{B}  ▶ Elige una opción: {R}").strip()

        if opcion == "1":
            params = pedir_params("paso1")
            correr("Paso 1 — MLP Base", "src/paso1_mlp_base.py", params)
            mostrar_menu()
        elif opcion == "2":
            params = pedir_params("paso2")
            correr("Paso 2 — GA Solo", "src/paso2_ga_solo.py", params)
            mostrar_menu()
        elif opcion == "3":
            params = pedir_params("paso3")
            correr("WolfGenetic — Lobos Grises", "src/paso3_gagw2_completo.py", params)
            mostrar_menu()
        elif opcion == "4":
            print(f"\n{AM}{B}  Modo producción — output en tiempo real{R}")
            print(f"{GR}  Ctrl+C cancela el paso actual y detiene todo{R}\n")
            params1 = pedir_params("paso1")
            if not correr("Paso 1 — MLP Base", "src/paso1_mlp_base.py", params1, prod=True):
                input(f"\n{GR}  Presiona Enter para volver al menú...{R}")
                mostrar_menu()
                continue
            params2 = pedir_params("paso2")
            if not correr("Paso 2 — GA Solo", "src/paso2_ga_solo.py", params2, prod=True):
                input(f"\n{GR}  Presiona Enter para volver al menú...{R}")
                mostrar_menu()
                continue
            params3 = pedir_params("paso3")
            correr("Paso 3 — WolfGenetic Completo", "src/paso3_gagw2_completo.py", params3, prod=True)
            input(f"\n{GR}  Presiona Enter para volver al menú...{R}")
            mostrar_menu()
        elif opcion == "0":
            os.system("clear")
            print(f"\n{VE}{B}  WolfGenetic finalizado.{R}\n")
            break
        else:
            print(f"\n{RO}  Opción no válida, intenta de nuevo.{R}")
            input(f"{GR}  Presiona Enter para continuar...{R}")
            mostrar_menu()


def mostrar_menu():
    os.system("clear")
    print(f"\n{AZ}{B}  ╔{'═'*43}╗{R}")
    print(f"{AZ}{B}  ║{'WolfGenetic — Menú Principal':^43}║{R}")
    print(f"{AZ}{B}  ║{'Dataset: Avila Bible':^43}║{R}")
    print(f"{AZ}{B}  ╚{'═'*43}╝{R}")
    print(f"\n{CY}  {'─'*43}{R}")
    print(f"  {B}{AM}[1]{R}  {B}Paso 1{R} — MLP Base")
    print(f"  {B}{AM}[2]{R}  {B}Paso 2{R} — GA Solo")
    print(f"  {B}{AM}[3]{R}  {B}WolfGenetic{R} — Lobos Grises solo")
    print(f"  {B}{AM}[4]{R}  {B}Correr todo{R}  (1 → 2 → 3)")
    print(f"  {B}{AM}[0]{R}  {RO}Salir{R}")
    print(f"{CY}  {'─'*43}{R}")


if __name__ == "__main__":
    menu()
