"""
ga_local_bin.py — Algoritmo Genético con codificación BINARIA (código Gray)

Versión binaria del GA. Cada individuo es una cadena de bits. Cada
hiperparámetro ocupa un bloque de bits y se decodifica con código Gray
para evitar el "Hamming cliff".

Garantía de límites: los bits SIEMPRE decodifican dentro de la ventana
[min, max] por construcción — no se necesita clipear. El entero que sale
de los bits está acotado en [0, 2^N-1], y al dividir entre 2^N-1 da un
porcentaje [0,1] que se proyecta dentro de la ventana.

Interfaz idéntica a ga_local.py para ser intercambiable con el GWO.
"""
import random

# Límites globales del hiperespacio (mismos que ga_local.py)
LB = [0.0001,   0.000001, 10,  1, 0, 150]
UB = [0.05,     0.1,      700, 4, 2, 500]
DIM = 6

# Bits asignados a cada dimensión:
#   [lr, alpha, neuronas, capas, activacion, max_iter]
BITS = [10, 10, 10, 3, 2, 8]
TOTAL_BITS = sum(BITS)


# ─── Conversiones Gray ↔ entero ───────────────────────────────────
def _gray_a_entero(gray):
    """Código Gray → binario normal → entero decimal."""
    binario = [gray[0]]
    for i in range(1, len(gray)):
        binario.append(binario[i - 1] ^ gray[i])  # XOR acumulativo
    entero = 0
    for bit in binario:
        entero = entero * 2 + bit
    return entero


def _entero_a_gray(entero, n_bits):
    """Entero decimal → binario → código Gray."""
    binario = [(entero >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]
    gray = [binario[0]]
    for i in range(1, n_bits):
        gray.append(binario[i - 1] ^ binario[i])
    return gray


# ─── Codificar / decodificar cromosoma completo ───────────────────
def _decodificar(cromosoma, win_lo, win_hi):
    """
    Cadena de bits → vector de 6 valores reales dentro de la ventana.
    win_lo / win_hi: límites por dimensión (los que define el lobo).
    """
    valores = []
    pos = 0
    for d in range(DIM):
        n = BITS[d]
        bloque = cromosoma[pos:pos + n]
        pos += n
        entero = _gray_a_entero(bloque)
        max_entero = (2 ** n) - 1
        # porcentaje [0,1] proyectado en la ventana → SIEMPRE dentro
        valor = win_lo[d] + (entero / max_entero) * (win_hi[d] - win_lo[d])
        valores.append(valor)
    return valores


def _codificar(valores, win_lo, win_hi):
    """Vector de reales → cadena de bits (inverso de _decodificar)."""
    cromosoma = []
    for d in range(DIM):
        n = BITS[d]
        rango = win_hi[d] - win_lo[d]
        if rango <= 0:
            entero = 0
        else:
            pct = (valores[d] - win_lo[d]) / rango
            pct = max(0.0, min(1.0, pct))
            entero = round(pct * ((2 ** n) - 1))
        cromosoma.extend(_entero_a_gray(entero, n))
    return cromosoma


# ─── Operadores genéticos sobre bits ──────────────────────────────
def _cromosoma_aleatorio():
    return [random.randint(0, 1) for _ in range(TOTAL_BITS)]


def _torneo(poblacion, scores, k=2):
    cand = random.sample(range(len(poblacion)), k)
    mejor = min(cand, key=lambda i: scores[i])
    return poblacion[mejor][:]


def _cruce_dos_puntos(p1, p2):
    """Cruce de dos puntos sobre la cadena de bits completa."""
    a, b = sorted(random.sample(range(1, TOTAL_BITS), 2))
    return p1[:a] + p2[a:b] + p1[b:]


def _mutar_bitflip(cromosoma, p_mutacion):
    """Voltea cada bit con probabilidad p_mutacion. Nunca sale de límites."""
    return [(1 - bit) if random.random() < p_mutacion else bit
            for bit in cromosoma]


# ─── GA binario principal ─────────────────────────────────────────
def ga_local_bin(func_objetivo, centro=None, radio=1.0,
                 lb=LB, ub=UB,
                 n_individuos=8, n_generaciones=3,
                 p_cruce=0.8, p_mutacion=0.05, paciencia=None):
    """
    GA con codificación binaria (Gray). Misma interfaz que ga_local.

    func_objetivo recibe un vector de 6 reales y devuelve el error.
    Devuelve (mejor_individuo_real, mejor_score, historial).

    Nota: p_mutacion en binario es por BIT (no por gen).
    paciencia: early stopping (generaciones sin mejora). None = sin él.
    """
    # Definir la ventana de búsqueda
    if centro is None:
        win_lo = list(lb)
        win_hi = list(ub)
    else:
        win_lo, win_hi = [], []
        for d in range(DIM):
            half = radio * (ub[d] - lb[d])
            win_lo.append(max(lb[d], centro[d] - half))
            win_hi.append(min(ub[d], centro[d] + half))

    # Población inicial de cromosomas (bits)
    poblacion = [_cromosoma_aleatorio() for _ in range(n_individuos)]
    if centro is not None:
        poblacion[0] = _codificar(centro, win_lo, win_hi)  # el lobo entra

    # Evaluar (decodificar → real → fitness)
    def fitness(crom):
        return func_objetivo(_decodificar(crom, win_lo, win_hi))

    scores = [fitness(c) for c in poblacion]
    mejor_idx = scores.index(min(scores))
    mejor_crom = poblacion[mejor_idx][:]
    mejor_score = scores[mejor_idx]
    historial = [mejor_score]

    sin_mejora = 0  # para early stopping

    for _ in range(n_generaciones):
        nueva_pob = [mejor_crom[:]]          # elitismo
        nueva_sco = [mejor_score]

        while len(nueva_pob) < n_individuos:
            p1 = _torneo(poblacion, scores)
            p2 = _torneo(poblacion, scores)
            hijo = _cruce_dos_puntos(p1, p2) if random.random() < p_cruce else p1[:]
            hijo = _mutar_bitflip(hijo, p_mutacion)
            nueva_pob.append(hijo)
            nueva_sco.append(fitness(hijo))

        poblacion, scores = nueva_pob, nueva_sco
        idx = scores.index(min(scores))
        if scores[idx] < mejor_score:
            mejor_score = scores[idx]
            mejor_crom = poblacion[idx][:]
            sin_mejora = 0
        else:
            sin_mejora += 1
        historial.append(mejor_score)

        if paciencia is not None and sin_mejora >= paciencia:
            break

    # Devolver el mejor como vector REAL (compatible con GWO)
    mejor_real = _decodificar(mejor_crom, win_lo, win_hi)
    return mejor_real, mejor_score, historial


if __name__ == "__main__":
    # ─── Pruebas unitarias del encoding ───────────────────────────
    print("=" * 55)
    print("  PRUEBAS — GA binario con código Gray")
    print("=" * 55)

    # Prueba 1: Gray ↔ entero es reversible
    print("\n[1] Reversibilidad Gray ↔ entero:")
    ok = True
    for val in [0, 1, 7, 255, 511, 1023]:
        gray = _entero_a_gray(val, 10)
        recuperado = _gray_a_entero(gray)
        estado = "OK" if recuperado == val else "FALLA"
        if recuperado != val:
            ok = False
        print(f"    {val:4d} → Gray → {recuperado:4d}  [{estado}]")

    # Prueba 2: código Gray — vecinos difieren en 1 bit
    print("\n[2] Propiedad Gray (vecinos difieren en 1 bit):")
    for val in [255, 256, 511, 512]:
        g1 = _entero_a_gray(val, 10)
        g2 = _entero_a_gray(val + 1, 10)
        difs = sum(b1 != b2 for b1, b2 in zip(g1, g2))
        estado = "OK" if difs == 1 else "FALLA"
        print(f"    {val} vs {val+1}: difieren en {difs} bit  [{estado}]")

    # Prueba 3: los bits NUNCA se salen de la ventana
    print("\n[3] Garantía de límites (1000 cromosomas aleatorios):")
    win_lo = [0.0001, 0.000001, 10, 1, 0, 150]
    win_hi = [0.05,   0.1,      700, 4, 2, 500]
    fuera = 0
    for _ in range(1000):
        crom = _cromosoma_aleatorio()
        vals = _decodificar(crom, win_lo, win_hi)
        for d in range(DIM):
            if not (win_lo[d] <= vals[d] <= win_hi[d]):
                fuera += 1
    print(f"    Valores fuera de límites: {fuera} / 6000  "
          f"[{'OK' if fuera == 0 else 'FALLA'}]")

    # Prueba 4: mutación masiva tampoco se sale
    print("\n[4] Mutación agresiva (p=0.5) no sale de límites:")
    fuera = 0
    for _ in range(1000):
        crom = _mutar_bitflip(_cromosoma_aleatorio(), 0.5)
        vals = _decodificar(crom, win_lo, win_hi)
        for d in range(DIM):
            if not (win_lo[d] <= vals[d] <= win_hi[d]):
                fuera += 1
    print(f"    Valores fuera de límites: {fuera} / 6000  "
          f"[{'OK' if fuera == 0 else 'FALLA'}]")

    # Prueba 5: ventana del lobo (radio chico) acota más
    print("\n[5] Ventana del lobo con radio chico:")
    centro = [0.01, 0.05, 350, 2, 1, 300]
    radio = 0.05
    w_lo, w_hi = [], []
    for d in range(DIM):
        half = radio * (UB[d] - LB[d])
        w_lo.append(max(LB[d], centro[d] - half))
        w_hi.append(min(UB[d], centro[d] + half))
    print(f"    Centro neuronas={centro[2]}, radio={radio}")
    print(f"    Ventana neuronas: [{w_lo[2]:.1f}, {w_hi[2]:.1f}]")
    crom = _cromosoma_aleatorio()
    vals = _decodificar(crom, w_lo, w_hi)
    dentro = w_lo[2] <= vals[2] <= w_hi[2]
    print(f"    Muestra aleatoria neuronas={vals[2]:.1f}  "
          f"[{'dentro' if dentro else 'FUERA'}]")

    print(f"\n{'='*55}")
    print(f"  Cromosoma: {TOTAL_BITS} bits = {BITS} por dimensión")
    print(f"  Todas las pruebas de encoding: {'PASARON' if ok and fuera == 0 else 'REVISAR'}")
