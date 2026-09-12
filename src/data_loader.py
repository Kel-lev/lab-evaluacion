"""
Módulo para la generación y carga del dataset sintético.
"""
import os
import pandas as pd
from sklearn.datasets import make_classification

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_PATH = os.path.join(DATA_DIR, "dataset.csv")


def generate_dataset(n_samples=1000, n_features=4, n_classes=3, random_state=42):
    """
    Genera un dataset sintético con make_classification.
    - 4 características continuas
    - Variable objetivo multiclase (3 clases)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])
    df["target"] = y
    return df


def save_dataset(df, path=DATA_PATH):
    """Guarda el dataset en formato CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[data_loader] Dataset guardado en: {path}")


def load_dataset(path=DATA_PATH):
    """Carga el dataset desde CSV. Si no existe, lo genera."""
    if not os.path.exists(path):
        print("[data_loader] Dataset no encontrado. Generando uno nuevo...")
        df = generate_dataset()
        save_dataset(df, path)
    else:
        df = pd.read_csv(path)
        print(f"[data_loader] Dataset cargado desde: {path}")
    return df


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    print(f"Shape: {df.shape}")
    print(f"Distribución de clases:\n{df['target'].value_counts()}")