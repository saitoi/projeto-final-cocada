import fasttext
import fasttext.util
import unicodedata
import string
import io
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import numpy as np

# Amostra selecionada foi o poema de Vinicius de Moraes
amostra = 'poema.txt'

def tokenizacao(texto):
    texto_sem_acentos = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto_sem_pontuacao = texto_sem_acentos.translate(str.maketrans('', '', string.punctuation))
    return texto_sem_pontuacao.lower().split()

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        palavra = tokens[0]
        vetor = list(map(float, tokens[1:]))
        data[palavra] = vetor
    return data

with open(f'amostras/{amostra}') as file:
    texto_original = file.read()

ft = fasttext.load_model('models/cc.pt.300.bin')
dimension = ft.get_dimension()

palavras_originais = tokenizacao(texto_original)
vetores_originais = [ft.get_word_vector(palavra) for palavra in palavras_originais]

vetores_criptografados = load_vectors('oov_vetores.txt')

pca = PCA(n_components=2)
vetores_reduzidos_originais = pca.fit_transform(vetores_originais)
vetores_reduzidos_criptografados = pca.transform(list(vetores_criptografados.values()))

def plot_clusters(vetores_originais, vetores_criptografados, method):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters_originais = kmeans.fit_predict(vetores_originais)
        clusters_criptografados = kmeans.predict(vetores_criptografados)
    elif method == 'dbscan':
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters_originais = dbscan.fit_predict(vetores_originais)
        clusters_criptografados = dbscan.fit_predict(vetores_criptografados)
    elif method == 'hierarchical':
        Z = linkage(vetores_originais, 'ward')
        plt.figure(figsize=(10, 6))
        dendrogram(Z)
        plt.title('Dendrograma dos Vetores de Palavras Originais')
        plt.xlabel('Índice das Palavras')
        plt.ylabel('Distância')
        plt.savefig('graficos/poema_decomposicao_pca_dendrograma.png')
        return
    else:
        raise ValueError("Método de clusterização não suportado.")

    plt.figure(figsize=(10, 6))
    plt.scatter(vetores_originais[:, 0], vetores_originais[:, 1], c=clusters_originais, cmap='viridis', label='Original')
    plt.scatter(vetores_criptografados[:, 0], vetores_criptografados[:, 1], c=clusters_criptografados, cmap='plasma', marker='x', label='Criptografado')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title(f'Clusterização {method.capitalize()} dos Vetores de Palavras Originais e Criptografados')
    plt.legend()
    plt.savefig(f'graficos/poema_decomposicao_pca_{method}.png')

# Escolha o método de clusterização: 'kmeans', 'dbscan' ou 'hierarchical'
plot_clusters(vetores_reduzidos_originais, vetores_reduzidos_criptografados, method='hierarchical')

