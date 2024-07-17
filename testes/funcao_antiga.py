%matplotlib widget
import fasttext
from base_funcoes import *

amostra = 'paragrafos'
metodo = 'cifra_cesar'
deslocamento = 15
titulo = 'Comparativo de Palavras Normais X Criptografadas com Cifra de César'

# # Inicializando Tokenização e Codificação
# leitura_impressao_dados(amostra, metodo=metodo, deslocamento=15)

# # Carregando o modelo
# ft = fasttext.load_model('models/cc.pt.300.bin')

# # Carregando os tokens originais e seus vetores
# palavras_originais = load_tokens(amostra)
# vetores_originais = [ft.get_word_vector(palavra) for palavra in palavras_originais]

# # Carregando os vetores criptografados
# vetores_criptografados = load_vectors(f'oov_vetores_{amostra}_{metodo}.txt')
# palavras_criptografadas = list(vetores_criptografados.keys())

# # Aplicação do PCA
# pca = PCA(n_components=3)
# vetores_reduzidos_originais = pca.fit_transform(vetores_originais)
# vetores_reduzidos_criptografados = pca.transform(list(vetores_criptografados.values()))

# plot_clusters(vetores_reduzidos_originais, vetores_reduzidos_criptografados, palavras_originais, palavras_criptografadas, method='kmeans', criptografia=metodo)
