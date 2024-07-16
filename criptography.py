from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Função para cifra de César
def caesar_cipher(text, shift):
    result = ""
    for i in range(len(text)):
        char = text[i]
        if char.isupper():
            result += chr((ord(char) + shift - 65) % 26 + 65)
        elif char.islower():
            result += chr((ord(char) + shift - 97) % 26 + 97)
        else:
            result += char
    return result

# Caminho para o modelo pré-treinado (substitua pelo caminho correto)
model_path = 'skip_s300.txt'

# Carregar o modelo pré-treinado
model = KeyedVectors.load_word2vec_format(model_path, binary=False)

# Texto de exemplo em português
text = """ 
De tudo, ao meu amor serei atento 
Antes, e com tal zelo, e sempre, e tanto 
Que mesmo em face do maior encanto 
Dele se encante mais meu pensamento. 
 
Quero vivê-lo em cada vão momento 
E em louvor hei de espalhar meu canto 
E rir meu riso e derramar meu pranto 
Ao seu pesar ou seu contentamento. 
 
E assim, quando mais tarde me procure 
Quem sabe a morte, angústia de quem vive 
Quem sabe a solidão, fim de quem ama 
 
Eu possa me dizer do amor (que tive): 
Que não seja imortal, posto que é chama 
Mas que seja infinito enquanto dure. 
"""

# Tokenização
tokens = text.lower().split()

# Obter vetores das palavras do modelo
vectors = []
words_in_model = []
for word in tokens:
    if word in model:
        vectors.append(model[word])
        words_in_model.append(word)

vectors = np.array(vectors)

# Aplicar PCA
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# Criptografar palavras
encrypted_words = [caesar_cipher(word, 3) for word in words_in_model]

# Obter vetores das palavras criptografadas
encrypted_vectors = []
valid_encrypted_words = []
for word in encrypted_words:
    if word in model:
        encrypted_vectors.append(model[word])
        valid_encrypted_words.append(word)
encrypted_vectors = np.array(encrypted_vectors)

# Aplicar PCA nos vetores criptografados
if len(encrypted_vectors) > 0:
    encrypted_reduced_vectors = pca.transform(encrypted_vectors)

# Visualização
plt.figure(figsize=(10, 7))

# Plotar palavras originais
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], color='blue', label='Original')
for i, word in enumerate(words_in_model):
    plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), color='blue')

# Plotar palavras criptografadas
if len(encrypted_vectors) > 0:
    plt.scatter(encrypted_reduced_vectors[:, 0], encrypted_reduced_vectors[:, 1], color='red', label='Criptografado')
    for i, word in enumerate(valid_encrypted_words):
        plt.annotate(word, (encrypted_reduced_vectors[i, 0], encrypted_reduced_vectors[i, 1]), color='red')

plt.legend()
plt.savefig('grafico_novo.png')
print("Gráfico salvo como 'word_embeddings_pca_cesar.png'")

