"""
Orquestador principal del pipeline de IA.
Ejecuta: carga de datos -> preprocesamiento -> supervisado -> no supervisado.
"""
import os
from src.data_loader import load_dataset
from src.preprocessing import split_data, scale_features
from src.supervised_model import train_classifier, evaluate_classifier
from src.unsupervised_model import apply_pca, apply_kmeans, plot_clusters


def main():
    print("\n" + "#" * 60)
    print("#  EXAMEN DE INTELIGENCIA ARTIFICIAL ")
    print("#" * 60 + "\n")

    # 1. Carga / generación del dataset
    df = load_dataset()

    # 2. Preprocesamiento
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    # 3. Pipeline supervisado
    model = train_classifier(X_train_s, y_train, model_type="logistic")
    metrics = evaluate_classifier(model, X_test_s, y_test)

    # 4. Pipeline no supervisado 
    X_all = df.drop(columns=["target"]).values
    X_all_s = scaler.transform(X_all)
    X_pca, var_acum, pca = apply_pca(X_all_s, n_components=2)
    labels, sil, kmeans = apply_kmeans(X_pca, k=3)
    plot_clusters(X_pca, labels, centroids=kmeans.cluster_centers_,
                  save_path="data/clusters_pca.png")

    print("\n" + "#" * 60)
    print("#  PIPELINE COMPLETADO EXITOSAMENTE")
    print("#" * 60)


if __name__ == "__main__":
    main()