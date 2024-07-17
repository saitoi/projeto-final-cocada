import fasttext
import fasttext.util
import unicodedata
import string
import io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Amostra selecionada foi um poema de Vinicius de Moraes
amostra = 'poema.txt'

def tokenizacao(texto):
    # Removendo acentos
    texto_sem_acentos = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

    # Removendo pontuação
    texto_sem_pontuacao = texto_sem_acentos.translate(str.maketrans('', '', string.punctuation))

    # Convertendo para minúsculas e dividindo em tokens
    return texto_sem_pontuacao.lower().split()

# Função para ler os vetores de palavras criptografadas
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        palavra = tokens[0]
        vetor = list(map(float, tokens[1:]))
        data[palavra] = vetor
    return data

# Acessando o texto original
with open(f'amostras/{amostra}') as file:
    texto_original = file.read()

# Carregando o modelo FastText
ft = fasttext.load_model('models/cc.pt.300.bin')
dimension = ft.get_dimension()

# Tokenizar e obter vetores de palavras originais
palavras_originais = tokenizacao(texto_original)
vetores_originais = [ft.get_word_vector(palavra) for palavra in palavras_originais]

# Carregar os vetores criptografados
vetores_criptografados = load_vectors('oov_vetores.txt')

print(f"Quantidade de palavras correspondidas: {len(vetores_originais)}")
print(f"Quantidade de palavras criptografradas correspondidas: {len(vetores_criptografados)}")
# Verificar se ambos têm a mesma quantidade de vetores
# if len(vetores_originais) != len(vetores_criptografados):
#     raise ValueError("A quantidade de vetores originais e criptografados não corresponde.")

# Aplicar PCA aos vetores originais
pca = PCA(n_components=2)
vetores_reduzidos_originais = pca.fit_transform(vetores_originais)

# Aplicar PCA aos vetores criptografados
vetores_reduzidos_criptografados = pca.transform(list(vetores_criptografados.values()))

# Plotar os dados
plt.figure(figsize=(10, 6))
plt.scatter(vetores_reduzidos_originais[:, 0], vetores_reduzidos_originais[:, 1], color='blue', label='Original')
plt.scatter(vetores_reduzidos_criptografados[:, 0], vetores_reduzidos_criptografados[:, 1], color='red', label='Criptografado')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA dos Vetores de Palavras Originais e Criptografados')
plt.legend()
plt.savefig('graficos/poema_decomposicao_pca_2.png')
