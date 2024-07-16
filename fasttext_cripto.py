import fasttext
import fasttext.util
import unicodedata
import string
import io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def tokenizacao(texto):
    def remover_acentos(texto):
        return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

    def remover_pontuacao(texto):
        return texto.translate(str.maketrans('', '', string.punctuation))

    texto_sem_acentos = remover_acentos(texto)
    texto_limpo = remover_pontuacao(texto_sem_acentos)
    return texto_limpo.split()


# Função para ler os vetores de palavras criptografadas
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


# Carregando os textos originais
# 'amostras/poema.txt'

# Carreagndo o modelo
ft = fasttext.load_model('models/cc.pt.300.bin')
dimension = ft.get_dimension()

# Carregando as palavras criptografadas
vetores_criptografados = load_vectors('oov_vectors.txt')

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)



