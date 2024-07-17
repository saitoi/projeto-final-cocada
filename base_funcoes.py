import io
import unicodedata
import string
import binascii
import fasttext
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP


# ---------------- Técnicas de Codificação ----------------

# Aplicação do cifra de césar resumida a uma única função
def cifra_cesar(tokens, deslocamento: int):
    texto_codificado = []
    deslocamento = int(deslocamento)
    for token in tokens:
        palavra_codificada = ""
        for caracter in token:
            if caracter.isupper():
                palavra_codificada += chr((int(ord(caracter)) + deslocamento - 65) % 26 + 65)
            elif caracter.islower():
                palavra_codificada += chr((int(ord(caracter)) + deslocamento - 97) % 26 + 97)
            else:
                palavra_codificada += caracter
        texto_codificado.append(palavra_codificada)
    return texto_codificado

   
def cifra_multiplicativa(tokens, chave: int):
    texto_codificado = []
    chave = int(chave)
    
    for token in tokens:
        palavra_codificada = ""
        for caracter in token:
            if caracter.isupper():
                palavra_codificada += chr(((ord(caracter) - 65) * chave % 26) + 65)
            elif caracter.islower():
                palavra_codificada += chr(((ord(caracter) - 97) * chave % 26) + 97)
            else:
                palavra_codificada += caracter
        texto_codificado.append(palavra_codificada)
    return texto_codificado

        
# Aplicação do RSA resumido a uma única função
def rsa_codificar(texto, semente=2032):
    chave = RSA.generate(semente)
    chave_publica = chave.publickey()

    cifra = PKCS1_OAEP.new(chave_publica)
    texto_codificado_final = []
    for palavra in texto:
        palavra_codificado = cifra.encrypt(palavra.encode('utf-8'))
        palavra_codificado_convertido = binascii.hexlify(palavra_codificado).decode('utf-8')
        texto_codificado_final.append(palavra_codificado_convertido)

    return texto_codificado_final 
    
# ---------------- Tratamento dos Dados ----------------

# Função para tratar as entradas de texto, antes de passarem pelo modelo
def tokenizacao(texto):
    # Removendo acentos
    texto_sem_acentos = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

    # Removendo pontuação
    texto_sem_pontuacao = texto_sem_acentos.translate(str.maketrans('', '', string.punctuation))

    return texto_sem_pontuacao.lower().split()


# Retorna um dicionário com as palavras criptografadas e seus respectivos vetores (na forma de listas)
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        palavra = tokens[0]
        vetor = list(map(float, tokens[1:]))
        data[palavra] = vetor
    return data


def leitura_impressao_dados(filename, deslocamento: int=3, metodo='cifra_cesar'):
    ft = fasttext.load_model('models/cc.pt.300.bin')
    
    with open(f'amostras/{filename}.txt', 'r') as input_file:
        texto_original = input_file.read()
        
        texto_tratado = tokenizacao(texto_original)
        
        texto_sem_repeticao = list(set(texto_tratado))
        
        # TODO: Verificar quais palavras estão no vocabulário do fasttext
        
        palavras_validas = []
        for palavra in texto_sem_repeticao:
            vector = ft.get_word_vector(palavra)
            if vector is not None:
                palavras_validas.append(palavra)
        
        if metodo == 'cifra_cesar':
            texto_criptografado = cifra_cesar(palavras_validas, deslocamento)
        elif metodo == 'cifra_multiplicativa':
            texto_criptografado = cifra_multiplicativa(palavras_validas, chave=deslocamento)
        else:
            texto_criptografado = rsa_codificar(palavras_validas)
        
        with open(f'tokens/{filename}_cripto_token_{metodo}.txt', 'w') as token_criptografado, \
             open(f'tokens/{filename}_token.txt', 'w') as token_normal:
                for token, cripto_token in zip(palavras_validas, texto_criptografado):
                    token_normal.write(token + '\n')
                    token_criptografado.write(cripto_token + '\n')
        
def load_tokens(filename):
    palavras_originais = []
    with open(f'tokens/{filename}_token.txt', 'r') as token_original:
        for token in token_original:
            palavras_originais.append(token.strip())
    return palavras_originais


# ---------------- Plotando os gráficos ----------------

def plot_clusters(vetores_originais, vetores_criptografados, palavras_originais, palavras_criptografadas, titulo, amostra, method='kmeans', criptografia='cifra_cesar'):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters_originais = kmeans.fit_predict(vetores_originais)
        clusters_criptografados = kmeans.predict(vetores_criptografados)
    else:
        raise ValueError("Método de clusterização não suportado.")

    fig = plt.figure(figsize=(15, 10))
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

    if criptografia == 'rsa': 
        for i, palavra in enumerate(palavras_criptografadas):
            ax.text(vetores_criptografados[i, 0], vetores_criptografados[i, 1], vetores_criptografados[i, 2],
                    f'c{i}', color='red', fontsize=7)
    else:
        for i, palavra in enumerate(palavras_criptografadas):
            ax.text(vetores_criptografados[i, 0], vetores_criptografados[i, 1], vetores_criptografados[i, 2],
                    palavra, color='red', fontsize=9)

    
    handle1 = mpatches.Patch(color='blue', label='Letra azul = Texto Original')
    handle2 = mpatches.Patch(color='red', label='Letra vermelha = Texto Criptografado')
    handle3 = mpatches.Patch(color='blue', label='Ponto Azul = Cluster 1')
    handle4 = mpatches.Patch(color='yellow', label='Ponto Amarelo = Cluster 2')

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title(titulo)
    ax.legend(handles=[handle1, handle2, handle3, handle4], loc='upper left', bbox_to_anchor=(1, 000000000))
    plt.subplots_adjust(right=0.75, bottom=0.2)
    plt.savefig(f'graficos/{amostra}_decomposicao_pca_{method}_3d.png')
    plt.show()

    # # Criando dicionários para mapear palavras por cluster
    # palavras_por_cluster_originais = {}
    # for palavra, cluster in zip(palavras_originais, clusters_originais):
    #     if cluster not in palavras_por_cluster_originais:
    #         palavras_por_cluster_originais[cluster] = []
    #     palavras_por_cluster_originais[cluster].append(palavra)

    # palavras_por_cluster_criptografadas = {}
    # for palavra, cluster in zip(palavras_criptografadas, clusters_criptografados):
    #     if cluster not in palavras_por_cluster_criptografadas:
    #         palavras_por_cluster_criptografadas[cluster] = []
    #     palavras_por_cluster_criptografadas[cluster].append(palavra)

    # return palavras_por_cluster_originais, palavras_por_cluster_criptografadas


def plot_clusters_2d(vetores_originais, vetores_criptografados, palavras_originais, palavras_criptografadas, titulo, amostra, method='kmeans', criptografia='cifra_cesar'):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters_originais = kmeans.fit_predict(vetores_originais)
        clusters_criptografados = kmeans.predict(vetores_criptografados)
    else:
        raise ValueError("Método de clusterização não suportado.")

    fig, ax = plt.subplots(figsize=(15, 10))

    # Scatter plot for original vectors
    scatter_orig = ax.scatter(vetores_originais[:, 0], vetores_originais[:, 1],
                              c=clusters_originais, cmap='plasma', marker='o', label='Original')

    # Scatter plot for encrypted vectors
    scatter_cript = ax.scatter(vetores_criptografados[:, 0], vetores_criptografados[:, 1],
                               c=clusters_criptografados, cmap='plasma', marker='x', label='Criptografado')

    # Adding text labels for each point
    for i, palavra in enumerate(palavras_originais):
        ax.text(vetores_originais[i, 0], vetores_originais[i, 1],
                palavra, color='blue', fontsize=9)

    if criptografia == 'rsa':
        for i, palavra in enumerate(palavras_criptografadas):
            ax.text(vetores_criptografados[i, 0], vetores_criptografados[i, 1],
                    f'c{i}', color='red', fontsize=7)
    else:
        for i, palavra in enumerate(palavras_criptografadas):
            ax.text(vetores_criptografados[i, 0], vetores_criptografados[i, 1],
                    palavra, color='red', fontsize=9)

    handle1 = mpatches.Patch(color='blue', label='Letra azul = Texto Original')
    handle2 = mpatches.Patch(color='red', label='Letra vermelha = Texto Criptografado')
    handle3 = mpatches.Patch(color='blue', label='Ponto Azul = Cluster 1')
    handle4 = mpatches.Patch(color='yellow', label='Ponto Amarelo = Cluster 2')

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title(titulo)
    ax.legend(handles=[handle1, handle2, handle3, handle4], loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.75, bottom=0.2)
    plt.savefig(f'graficos/{amostra}_decomposicao_pca_{method}_2d.png')
    plt.show()

def plot_clusters_w(vetores_originais, vetores_criptografados, palavras_originais, palavras_criptografadas, titulo, amostra, method='kmeans', criptografia='cifra_cesar'):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters_originais = kmeans.fit_predict(vetores_originais)
        clusters_criptografados = kmeans.predict(vetores_criptografados)
    else:
        raise ValueError("Método de clusterização não suportado.")

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for original vectors
    scatter_orig = ax.scatter(vetores_originais[:, 0], vetores_originais[:, 1], vetores_originais[:, 2],
                              c=clusters_originais, cmap='Blues', marker='o', label='Original')

    # Scatter plot for encrypted vectors
    scatter_crypt = ax.scatter(vetores_criptografados[:, 0], vetores_criptografados[:, 1], vetores_criptografados[:, 2],
                               c=clusters_criptografados, cmap='Oranges', marker='x', label='Criptografado')

    # Adding text labels for each point
    for i, palavra in enumerate(palavras_originais):
        ax.text(vetores_originais[i, 0], vetores_originais[i, 1], vetores_originais[i, 2],
                palavra, color='blue', fontsize=9)

    for i, palavra in enumerate(palavras_criptografadas):
        ax.text(vetores_criptografados[i, 0], vetores_criptografados[i, 1], vetores_criptografados[i, 2],
                palavra, color='red', fontsize=9)

    # Adicionando o plano formado pelos três componentes principais
    pca = PCA(n_components=3)
    pca.fit(vetores_originais)
    pc1, pc2, pc3 = pca.components_

    # Create a grid to plot the plane
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
    
    if pc1[2] != 0:  # Check to prevent division by zero
        Z = (-pc1[0] * X - pc1[1] * Y) / pc1[2]
        ax.plot_surface(X, Y, Z, color='gray', alpha=0.5, rstride=100, cstride=100)
    else:
        print("Warning: Division by zero encountered in the plane calculation")

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title(titulo)
    ax.legend(loc='best')
    plt.savefig(f'graficos/{amostra}_decomposicao_pca_{method}_3d.png')
    plt.show()

# ---------------- Função Faz Tudo ----------------

def processamento_plotting(amostra, metodo, deslocamento, titulo):
    leitura_impressao_dados(amostra, deslocamento, metodo=metodo)

    ft = fasttext.load_model('models/cc.pt.300.bin')

    palavras_originais = load_tokens(amostra)
    vetores_originais = [ft.get_word_vector(palavra) for palavra in palavras_originais]
    vetores_criptografados = load_vectors(f'oov_vetores_{amostra}_{metodo}.txt')
    palavras_criptografadas = list(vetores_criptografados.keys())

    pca = PCA(n_components=3)
    vetores_reduzidos_originais = pca.fit_transform(vetores_originais)
    vetores_reduzidos_criptografados = pca.transform(list(vetores_criptografados.values()))
    plot_clusters(vetores_reduzidos_originais, vetores_reduzidos_criptografados, palavras_originais, palavras_criptografadas, amostra=amostra, method='kmeans', criptografia=metodo, titulo=titulo)


    autovetores = pca.components_
    vetores_criptografados_proj = np.dot(list(vetores_criptografados.values()), autovetores.T)

    tamanhos_projecoes = np.linalg.norm(vetores_criptografados_proj, axis=1)

    return vetores_reduzidos_originais, vetores_reduzidos_criptografados, tamanhos_projecoes

def processamento_plotting_2d(amostra, metodo, deslocamento, titulo):
    leitura_impressao_dados(amostra, deslocamento, metodo=metodo)

    ft = fasttext.load_model('models/cc.pt.300.bin')

    palavras_originais = load_tokens(amostra)
    vetores_originais = [ft.get_word_vector(palavra) for palavra in palavras_originais]
    vetores_criptografados = load_vectors(f'oov_vetores_{amostra}_{metodo}.txt')
    palavras_criptografadas = list(vetores_criptografados.keys())

    pca = PCA(n_components=2)  # Usando 2 componentes principais
    vetores_reduzidos_originais = pca.fit_transform(vetores_originais)
    vetores_reduzidos_criptografados = pca.transform(list(vetores_criptografados.values()))
    plot_clusters_2d(vetores_reduzidos_originais, vetores_reduzidos_criptografados, palavras_originais, palavras_criptografadas, amostra=amostra, method='kmeans', criptografia=metodo, titulo=titulo)

    autovetores = pca.components_
    vetores_criptografados_proj = np.dot(list(vetores_criptografados.values()), autovetores.T)

    tamanhos_projecoes = np.linalg.norm(vetores_criptografados_proj, axis=1)

    return vetores_reduzidos_originais, vetores_reduzidos_criptografados, tamanhos_projecoes
