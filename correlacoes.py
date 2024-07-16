import numpy as np
from scipy.stats import pearsonr

# Dicionário para armazenar correlações
correlacoes = {}

# Assumindo que `vetores_originais` e `vetores_criptografados` são dicionários com palavras como chaves e vetores como valores
for palavra, vetor_original in vetores_originais.items():
    vetor_criptografado = vetores_criptografados.get(palavra)
    if vetor_criptografado is not None:
        # Calcular a correlação de Pearson entre os dois vetores
        correlacao, _ = pearsonr(vetor_original, vetor_criptografado)
        correlacoes[palavra] = correlacao

# Imprimir as correlações
for palavra, correlacao in correlacoes.items():
    print(f"Correlação entre vetores original e criptografado para '{palavra}': {correlacao:.2f}")
