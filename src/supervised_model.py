"""
Pipeline supervisado: entrenamiento y evaluación de un clasificador.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def train_classifier(X_train, y_train, model_type="logistic"):
    """
    Entrena un clasificador supervisado.
    Opciones: 'logistic', 'tree', 'svm'
    """
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "tree":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "svm":
        from sklearn.svm import SVC
        model = SVC(kernel="rbf", random_state=42)
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")

    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test, y_test):
    """
    Calcula Accuracy, F1-score y Matriz de Confusión.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 50)
    print("  EVALUACIÓN DEL MODELO SUPERVISADO")
    print("=" * 50)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nMatriz de Confusión:")
    print(cm)
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    return {"accuracy": acc, "f1_score": f1, "confusion_matrix": cm}