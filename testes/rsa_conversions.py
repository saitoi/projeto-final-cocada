from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import binascii
import unicodedata
import string

def remover_acentos(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

def remover_pontuacao(texto):
    return texto.translate(str.maketrans('', '', string.punctuation))

# Remover acentos e pontuações
texto = """
De tudo, ao meu amor serei atento
Antes, e com tal zelo, e sempre, e tanto
Que mesmo em face do maior encanto
Dele se encante mais meu pensamento.
"""

texto_sem_acentos = remover_acentos(texto)
texto_limpo = remover_pontuacao(texto_sem_acentos)

# Gerar chaves RSA
key = RSA.generate(2048)
public_key = key.publickey()
cipher = PKCS1_OAEP.new(public_key)

# Converter o texto em bytes e criptografar
texto_bytes = texto_limpo.encode('utf-8')
cipher_text = cipher.encrypt(texto_bytes)

# Converter o texto cifrado em uma forma legível (hexadecimal)
cipher_text_hex = binascii.hexlify(cipher_text).decode('utf-8')

print(cipher_text_hex)

