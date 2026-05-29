import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
_TEST_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "test.csv")


def cargar_datos(val_size=0.2, random_state=42):
    """
    Carga Pendigits desde los splits originales de UCI y separa validación.

    Columnas del dataset:
        X → x1..x8, y1..y8  (coordenadas del trazo del dígito, valores en [0,100])
        y → clase            (dígito real: 0-9)

    Splits resultantes:
        Train (5,995) → entrena el MLP dentro de funcion_objetivo
        Val   (1,499) → evalúa fitness (GA y GWO nunca ven el test)
        Test  (3,498) → solo se usa al final para reportar accuracy final

    Returns:
        X_train, X_val, X_test : arrays normalizados (StandardScaler)
        y_train, y_val, y_test : etiquetas (enteros 0-9)
    """
    df_train = pd.read_csv(_TRAIN_PATH)
    df_test  = pd.read_csv(_TEST_PATH)

    X_tv   = df_train.drop(columns=["clase"]).values.astype(float)
    y_tv   = df_train["clase"].values.astype(int)
    X_test = df_test.drop(columns=["clase"]).values.astype(float)
    y_test = df_test["clase"].values.astype(int)

    # Separar validación del train original (20% estratificado)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_size,
        random_state=random_state,
        stratify=y_tv
    )

    # Normalizar: el scaler aprende SOLO del train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) #aprende y calcula 
    X_val   = scaler.transform(X_val) # solo aplica lo calculado 
    X_test  = scaler.transform(X_test) # solo aplica lo calculado 

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = cargar_datos()

    print("Split de datos:")
    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")
    print(f"\nTotal : {len(y_train) + len(y_val) + len(y_test)} instancias")
    print(f"\nMedia  post-scaler (train): {X_train.mean():.6f}  (debe ser ≈ 0)")
    print(f"Std   post-scaler (train): {X_train.std():.6f}   (debe ser ≈ 1)")
