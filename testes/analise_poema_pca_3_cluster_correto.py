import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

def plot_clusters(vetores_originais, vetores_criptografados, palavras_originais, palavras_criptografadas, method):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters_originais = kmeans.fit_predict(vetores_originais)
        clusters_criptografados = kmeans.predict(vetores_criptografados)
    elif method == 'dbscan':
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters_originais = dbscan.fit_predict(vetores_originais)
        clusters_criptografados = dbscan.fit_predict(vetores_criptografados)
    else:
        raise ValueError("Método de clusterização não suportado.")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    scatter_orig = ax.scatter(vetores_originais[:, 0], vetores_originais[:, 1], vetores_originais[:, 2], c=clusters_originais, cmap='viridis', label='Original')
    scatter_cripto = ax.scatter(vetores_criptografados[:, 0], vetores_criptografados[:, 1], vetores_criptografados[:, 2], c=clusters_criptografados, cmap='viridis', marker='x', label='Criptografado')

    for i, palavra in enumerate(palavras_originais):
        ax.text(vetores_originais[i, 0], vetores_originais[i, 1], vetores_originais[i, 2], palavra, color=plt.cm.viridis(clusters_originais[i] / (max(clusters_originais) + 1)), fontsize=9)
    
    for i, palavra in enumerate(palavras_criptografadas):
        ax.text(vetores_criptografados[i, 0], vetores_criptografados[i, 1], vetores_criptografados[i, 2], palavra, color=plt.cm.viridis(clusters_criptografados[i] / (max(clusters_criptografados) + 1)), fontsize=9)

    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.set_zlabel('Componente Principal 3')
    ax.set_title(f'Clusterização {method.capitalize()} dos Vetores de Palavras Originais e Criptografados')
    ax.legend()
    plt.savefig(f'graficos/poema_decomposicao_pca_{method}_3d.png')
    plt.show()

    palavras_por_cluster_originais = {}
    for palavra, cluster in zip(palavras_originais, clusters_originais):
        if cluster not in palavras_por_cluster_originais:
            palavras_por_cluster_originais[cluster] = []
        palavras_por_cluster_originais[cluster].append(palavra)

    palavras_por_cluster_criptografadas = {}
    for palavra, cluster in zip(palavras_criptografadas, clusters_criptografados):
        if cluster not in palavras_por_cluster_criptografadas:
            palavras_por_cluster_criptografadas[cluster] = []
        palavras_por_cluster_criptografadas[cluster].append(palavra)

    return palavras_por_cluster_originais, palavras_por_cluster_criptografadas

# Exemplo de uso
palavras_originais = ['palavra1', 'palavra2', 'palavra3']  # Substitua com suas palavras reais
palavras_criptografadas = ['palavra4', 'palavra5', 'palavra6']  # Substitua com suas palavras reais

# Escolha o método de clusterização: 'kmeans' ou 'dbscan'
resultados = plot_clusters(vetores_reduzidos_originais, vetores_reduzidos_criptografados, palavras_originais, palavras_criptografadas, method='kmeans')

# Listar palavras em cada agrupamento
palavras_por_cluster_originais, palavras_por_cluster_criptografadas = resultados

print("Palavras Originais por Cluster:")
for cluster, palavras in palavras_por_cluster_originais.items():
    print(f"Cluster {cluster}: {palavras}")

print("\nPalavras Criptografadas por Cluster:")
for cluster, palavras in palavras_por_cluster_criptografadas.items():
    print(f"Cluster {cluster}: {palavras}")
