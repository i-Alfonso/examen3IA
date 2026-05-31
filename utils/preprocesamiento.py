import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "data2", "avila", "avila-tr.txt")
_TEST_PATH  = os.path.join(os.path.dirname(__file__), "..", "data2", "avila", "avila-ts.txt")

_le = LabelEncoder()


def cargar_datos(val_size=0.2, random_state=42):
    """
    Carga Avila Bible desde los splits originales de UCI y separa validación.

    Columnas del dataset:
        X → F1..F10  (características paleográficas, ya Z-normalizadas)
        y → clase    (copista: A,B,C,D,E,F,G,H,I,W,X,Y → codificado 0-11)

    Notas:
        - StandardScaler aplicado para comprimir outliers (F2 max=386, F6, F7, F9)
        - Clase B tiene solo 5 instancias → split estratificado es crítico
        - Desbalance extremo: A=4286, B=5 → accuracy base esperado ~70-80%

    Splits resultantes:
        Train (8,344) → entrena el MLP dentro de funcion_objetivo
        Val   (2,086) → evalúa fitness (GA y GWO nunca ven el test)
        Test (10,437) → solo se usa al final para reportar accuracy final

    Returns:
        X_train, X_val, X_test : arrays normalizados (StandardScaler)
        y_train, y_val, y_test : etiquetas (enteros 0-11)
    """
    df_train = pd.read_csv(_TRAIN_PATH, header=None)
    df_test  = pd.read_csv(_TEST_PATH,  header=None)

    X_tv   = df_train.iloc[:, :-1].values.astype(float)
    y_tv   = df_train.iloc[:, -1].values
    X_test = df_test.iloc[:, :-1].values.astype(float)
    y_test = df_test.iloc[:, -1].values

    # Codificar clases (letras → enteros)
    _le.fit(np.concatenate([y_tv, y_test]))
    y_tv   = _le.transform(y_tv).astype(int)
    y_test = _le.transform(y_test).astype(int)

    # Separar validación (estratificado — crítico por clase B con 5 instancias)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_size,
        random_state=random_state,
        stratify=y_tv
    )

    # StandardScaler para comprimir outliers (F2 max=386, F6, F7, F9)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # aprende y calcula
    X_val   = scaler.transform(X_val)        # solo aplica lo calculado
    X_test  = scaler.transform(X_test)       # solo aplica lo calculado

    return X_train, X_val, X_test, y_train, y_val, y_test


def clases_nombres():
    """Retorna los nombres originales de las clases (A, B, C...)."""
    return list(_le.classes_)


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = cargar_datos()

    print("Dataset: Avila Bible")
    print(f"\nSplit de datos:")
    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")
    print(f"\nTotal : {len(y_train) + len(y_val) + len(y_test)} instancias")
    print(f"\nClases: {clases_nombres()}")
    print(f"\nMedia  post-scaler (train): {X_train.mean():.6f}  (debe ser ≈ 0)")
    print(f"Std   post-scaler (train): {X_train.std():.6f}   (debe ser ≈ 1)")
    print(f"\nDistribución train:")
    for i, nombre in enumerate(clases_nombres()):
        n = (y_train == i).sum()
        print(f"  {nombre}: {n}")
