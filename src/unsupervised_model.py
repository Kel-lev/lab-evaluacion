"""
Pipeline no supervisado: PCA + K-Means + Silueta.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def apply_pca(X, n_components=2):
    """
    Reduce las 4 características a 2 componentes principales.
    Retorna: X_pca, varianza explicada acumulada, modelo PCA.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    var_acum = np.cumsum(pca.explained_variance_ratio_)
    print("=" * 50)
    print("  ANÁLISIS PCA")
    print("=" * 50)
    print(f"Varianza explicada por componente: {pca.explained_variance_ratio_}")
    print(f"Varianza explicada acumulada    : {var_acum}")
    return X_pca, var_acum, pca


def apply_kmeans(X_pca, k=3, random_state=42):
    """
    Aplica K-Means con k clusters sobre las componentes principales.
    Retorna: etiquetas, silhouette score, modelo.
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    print("=" * 50)
    print("  AGRUPAMIENTO K-MEANS")
    print("=" * 50)
    print(f"k = {k}")
    print(f"Coeficiente de Silueta: {sil:.4f}")
    return labels, sil, kmeans


def plot_clusters(X_pca, labels, centroids=None, save_path=None):
    """
    Visualización 2D de los clusters.
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=40, alpha=0.7
    )
    if centroids is not None:
        plt.scatter(
            centroids[:, 0], centroids[:, 1],
            c="red", marker="X", s=200, label="Centroides"
        )
    plt.title("Clusters K-Means sobre componentes PCA (2D)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend()
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[unsupervised_model] Figura guardada en: {save_path}")
    plt.show()