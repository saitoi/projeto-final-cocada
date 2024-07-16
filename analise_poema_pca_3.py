import fasttext
import fasttext.util
import unicodedata
import string
import io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        palavra = tokens[0]
        vetor = list(map(float, tokens[1:]))
        data[palavra] = vetor
    return data

# Carregar o modelo FastText
ft = fasttext.load_model('models/cc.pt.300.bin')
dimension = ft.get_dimension()

# Texto original (exemplo)
texto_original = """
De tudo, ao meu amor serei atento
Antes, e com tal zelo, e sempre, e tanto
Que mesmo em face do maior encanto
Dele se encante mais meu pensamento.

Quero vive-lo em cada vao momento
E em louvor hei de espalhar meu canto
E rir meu riso e derramar meu pranto
Ao seu pesar ou seu contentamento.

E assim, quando mais tarde me procure
Quem sabe a morte, angustia de quem vive
Quem sabe a solidao, fim de quem ama

Eu possa me dizer do amor que tive
Que nao seja imortal, posto que e chama
Mas que seja infinito enquanto dure.
"""

# Tokenizar e obter vetores de palavras originais
palavras_originais = tokenizacao(texto_original)
vetores_originais = [ft.get_word_vector(palavra) for palavra in palavras_originais]

# Carregar os vetores criptografados
vetores_criptografados = load_vectors('oov_vetores.txt')

print(f"Quantidade de palavras correspondidas: {len(vetores_originais)}")
print(f"Quantidade de palavras criptografradas correspondidas: {len(vetores_criptografados)}")

# Aplicar PCA aos vetores originais
pca = PCA(n_components=3)
vetores_reduzidos_originais = pca.fit_transform(vetores_originais)

# Aplicar PCA aos vetores criptografados
vetores_reduzidos_criptografados = pca.transform(list(vetores_criptografados.values()))

# Plotar os dados em 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vetores_reduzidos_originais[:, 0], vetores_reduzidos_originais[:, 1], vetores_reduzidos_originais[:, 2], color='blue', label='Original')
ax.scatter(vetores_reduzidos_criptografados[:, 0], vetores_reduzidos_criptografados[:, 1], vetores_reduzidos_criptografados[:, 2], color='red', label='Criptografado')

# Adicionar labels aos pontos originais
for i, palavra in enumerate(palavras_originais):
    ax.text(vetores_reduzidos_originais[i, 0], vetores_reduzidos_originais[i, 1], vetores_reduzidos_originais[i, 2], palavra, color='blue', fontsize=9)

# Adicionar labels aos pontos criptografados
for i, palavra in enumerate(vetores_criptografados.keys()):
    ax.text(vetores_reduzidos_criptografados[i, 0], vetores_reduzidos_criptografados[i, 1], vetores_reduzidos_criptografados[i, 2], palavra, color='red', fontsize=9)

ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_zlabel('Componente Principal 3')
ax.set_title('PCA dos Vetores de Palavras Originais e Criptografados')
ax.legend()
plt.savefig('graficos/poema_decomposicao_pca_3d.png')
plt.show()
