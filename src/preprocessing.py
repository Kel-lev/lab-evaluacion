"""
Módulo de preprocesamiento: escalado con StandardScaler.
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def split_data(df, test_size=0.2, random_state=42):
    """
    Divide el dataset en Train/Test (80/20).
    Retorna: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Ajusta StandardScaler SOLO en train y transforma ambos conjuntos.
    Evita data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler