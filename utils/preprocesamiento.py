"""
preprocesamiento.py — Carga del dataset Avila y construcción de la muestra

Responsabilidad única: leer Avila, escalarlo y partirlo en train/val/test.
Todo lo que viene después (red, GA, GWO) recibe los datos desde aquí.

El split y el escalador se controlan desde config.yaml.
"""
import os
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from utils.config import CONFIG

# Rutas a los dos archivos originales de Avila (UCI)
_DIR = os.path.join(os.path.dirname(__file__), "..", "data2", "avila")
_TR  = os.path.join(_DIR, "avila-tr.txt")
_TS  = os.path.join(_DIR, "avila-ts.txt")

# LabelEncoder compartido (letras A-Y → enteros 0-11)
_le = LabelEncoder()


def cargar_datos():
    """
    Carga Avila y devuelve el split de 3 vías ya escalado.

    Proceso:
      1. Junta los dos archivos originales (avila-tr + avila-ts = 20,867 filas).
      2. Codifica las clases (letras de copista → enteros).
      3. Hace un split aleatorio ESTRATIFICADO según las proporciones
         de config.yaml (por defecto 70/15/15). Estratificar es clave
         por el fuerte desbalance del dataset (la clase B tiene ~10 casos).
      4. Escala con RobustScaler (mediana + IQR), que resiste los valores
         atípicos extremos que tiene Avila en varias columnas.
         El escalador se ajusta SOLO con train y se aplica a val y test,
         para no filtrar información entre conjuntos.

    Returns:
        X_train, X_val, X_test : arrays escalados
        y_train, y_val, y_test : etiquetas (enteros 0-11)
    """
    cfg    = CONFIG["dataset"]
    seed   = cfg["random_state"]
    p_test = cfg["test"]
    p_val  = cfg["val"]

    # 1-2. Cargar y juntar ambos archivos, codificar clases
    df = pd.concat([
        pd.read_csv(_TR, header=None),
        pd.read_csv(_TS, header=None)
    ], ignore_index=True)
    X = df.iloc[:, :-1].values.astype(float)
    y = _le.fit_transform(df.iloc[:, -1].values)

    # 3. Split estratificado en dos pasos
    #    Paso A: separar test
    X_resto, X_test, y_resto, y_test = train_test_split(
        X, y, test_size=p_test, random_state=seed, stratify=y)
    #    Paso B: separar val del resto (proporción relativa al resto)
    val_rel = p_val / (1.0 - p_test)
    X_train, X_val, y_train, y_val = train_test_split(
        X_resto, y_resto, test_size=val_rel, random_state=seed, stratify=y_resto)

    # 4. Escalado (robust o standard según config)
    scaler = RobustScaler() if cfg["scaler"] == "robust" else StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def clases_nombres():
    """Nombres originales de las clases (las letras de los copistas)."""
    return list(_le.classes_)


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = cargar_datos()
    print(f"Dataset: {CONFIG['dataset']['nombre']}")
    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")
    print(f"  Total : {len(y_train) + len(y_val) + len(y_test)} instancias")
    print(f"  Clases: {clases_nombres()}")
    print(f"  Escalador: {CONFIG['dataset']['scaler']}")
    print(f"\n  Media train post-escala: {X_train.mean():.4f}")
