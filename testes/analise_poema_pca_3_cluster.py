import numpy as np
import fasttext
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from base_funcoes import leitura_impressao_dados, load_vectors, load_tokens

amostra = 'poema'

# Amostra selecionada foi o peoma de Vinicius de Moraes
# Carregamento => Tokenização => Codificação
leitura_impressao_dados(amostra, deslocamento=3)

# Carregar o modelo FastText
ft = fasttext.load_model('models/cc.pt.300.bin')
dimension = ft.get_dimension()

# TODO: Preciso remover palavras criptografadas daqui
palavras_originais = load_tokens(amostra)

# Selecionando todos os vetores de palavras originais em ordem
vetores_originais = [ft.get_word_vector(palavra) for palavra in palavras_originais]
vetores_criptografados = load_vectors('oov_vetores.txt') # 

palavras_criptografadas = list(vetores_criptografados.keys())

# Dicionário de correspondência {palavra_original: criptografada}
correspondencia = {}
for i, criptografada in enumerate(palavras_criptografadas):
    correspondencia[palavras_originais[i]] = criptografada
    
# Carregar os vetores criptografados
# Retorna um dicionário com as palavras criptografadas e seus respectivos vetores

print(f"Quantidade de palavras originais: {len(palavras_originais)}")
print(f"Quantidade de palavras criptografadas correspondidas: {len(palavras_criptografadas)}")

# Aplicar PCA aos vetores originais
pca = PCA(n_components=3)
vetores_reduzidos_originais = pca.fit_transform(vetores_originais)

# Aplicar PCA aos vetores criptografados
vetores_reduzidos_criptografados = pca.transform(list(vetores_criptografados.values()))

# def plot_clusters(vetores_originais, vetores_criptografados, method):
#     if method == 'kmeans':
#         kmeans = KMeans(n_clusters=2, random_state=42)
#         clusters_originais = kmeans.fit_predict(vetores_originais)
#         clusters_criptografados = kmeans.predict(vetores_criptografados)
#     elif method == 'dbscan':
#         dbscan = DBSCAN(eps=0.5, min_samples=5)
#         clusters_originais = dbscan.fit_predict(vetores_originais)
#         clusters_criptografados = dbscan.fit_predict(vetores_criptografados)
#     else:
#         raise ValueError("Método de clusterização não suportado.")

#     fig = plt.figure(figsize=(10, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(vetores_originais[:, 0], vetores_originais[:, 1], vetores_originais[:, 2], c=clusters_originais, cmap='viridis', label='Original')
#     ax.scatter(vetores_criptografados[:, 0], vetores_criptografados[:, 1], vetores_criptografados[:, 2], c=clusters_criptografados, cmap='plasma', marker='x', label='Criptografado')

#     for i, palavra in enumerate(palavras_originais):
#         ax.text(vetores_originais[i, 0], vetores_originais[i, 1], vetores_originais[i, 2], palavra, color='blue', fontsize=9)
    
#     for i, palavra in enumerate(palavras_criptografadas):
#         ax.text(vetores_reduzidos_criptografados[i, 0], vetores_reduzidos_criptografados[i, 1], vetores_reduzidos_criptografados[i, 2], palavra, color='red', fontsize=9)

#     ax.set_xlabel('Componente Principal 1')
#     ax.set_ylabel('Componente Principal 2')
#     ax.set_zlabel('Componente Principal 3')
#     ax.set_title(f'Clusterização {method.capitalize()} dos Vetores de Palavras Originais e Criptografados')
#     ax.legend()
#     plt.savefig(f'graficos/poema_decomposicao_pca_{method}_3d.png')
#     plt.show()

def plot_clusters(vetores_originais, vetores_criptografados, palavras_originais, palavras_criptografadas, method='kmeans'):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters_originais = kmeans.fit_predict(vetores_originais)
        clusters_criptografados = kmeans.predict(vetores_criptografados)
    else:
        raise ValueError("Método de clusterização não suportado.")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Combine all vectors and clusters for unified color mapping
    # all_vectors = np.vstack((vetores_originais, vetores_criptografados))
    # all_clusters = np.concatenate((clusters_originais, clusters_criptografados))

    # Scatter plot for original vectors
    ax.scatter(vetores_originais[:, 0], vetores_originais[:, 1], vetores_originais[:, 2],
               c=clusters_originais, cmap='plasma', marker='o', label='Original')

    # Scatter plot for encrypted vectors
    ax.scatter(vetores_criptografados[:, 0], vetores_criptografados[:, 1], vetores_criptografados[:, 2],
               c=clusters_criptografados, cmap='plasma', marker='x', label='Criptografado')

    # Adding text labels for each point
    for i, palavra in enumerate(palavras_originais):
        ax.text(vetores_originais[i, 0], vetores_originais[i, 1], vetores_originais[i, 2],
                palavra, color='blue', fontsize=9)
    
    for i, palavra in enumerate(palavras_criptografadas):
        ax.text(vetores_criptografados[i, 0], vetores_criptografados[i, 1], vetores_criptografados[i, 2],
                palavra, color='red', fontsize=9)

    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.set_zlabel('Componente Principal 3')
    ax.set_title(f'Clusterização {method.capitalize()} dos Vetores de Palavras Originais e Criptografados')
    ax.legend()
    plt.savefig(f'graficos/poema_decomposicao_pca_{method}_3d.png')
    plt.show()

    # Criando dicionários para mapear palavras por cluster
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

# Escolha o método de clusterização: 'kmeans' ou 'dbscan'
palavras_originais_cluster, palavras_criptografadas_cluster = plot_clusters(vetores_reduzidos_originais, vetores_reduzidos_criptografados, palavras_originais, palavras_criptografadas, method='kmeans')

# print("Palavras Originais por Cluster:")
# for cluster, palavras in palavras_originais_cluster.items():
#     print(f"Cluster {cluster}: {palavras}")

# print("\nPalavras Criptografadas por Cluster:")
# for cluster, palavras in palavras_criptografadas_cluster.items():
#     print(f"Cluster {cluster}: {palavras}")

def contar_correspondencias_criptografadas(palavras_por_cluster_originais, palavras_por_cluster_criptografadas, correspondencia):
    contagem_correspondencias = {}

    # Iterar sobre cada cluster de palavras originais
    for cluster, palavras_originais in palavras_por_cluster_originais.items():
        if cluster in palavras_por_cluster_criptografadas:
            # Obter as palavras criptografadas para o mesmo cluster
            palavras_criptografadas_cluster = palavras_por_cluster_criptografadas[cluster]
            # Criar um contador para as correspondências criptografadas neste cluster
            contagem_cluster = {}
            for palavra_orig in palavras_originais:
                palavra_cripto = correspondencia.get(palavra_orig, None)
                if palavra_cripto in palavras_criptografadas_cluster:
                    if palavra_cripto not in contagem_cluster:
                        contagem_cluster[palavra_cripto] = 0
                    contagem_cluster[palavra_cripto] += 1
            contagem_correspondencias[cluster] = contagem_cluster

    return contagem_correspondencias

# Uso da função
resultado_contagem = contar_correspondencias_criptografadas(palavras_originais_cluster, palavras_criptografadas_cluster, correspondencia)

print(resultado_contagem)
