# Instruções para executar `Relatorio Final Cocada`

## Baixar e Extrair o Modelo FastText

- Baixe o modelo usando esse link: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.bin.gz

- Extraia o modelo fazendo: `gunzip diretorio/para/modelo`.

- Coloque o modelo no diretorio `models/` dentro do seu projeto.

## Instale os Requisitos

- Crie um ambiente virtual `.venv`, baixando os requerimentos com `pip3 install -r requirements.txt`.

- Utilizar o ambiente via comandos: **UNIX** `source .venv/bin/activate` && **Windowns** `Scripts/activate`

- Instale requisitos adicionais para o Jupyter Notebook usando `pip install` para os seguintes pacotes:

    - `pycryptodome`

    - `ipyml`

- Prontinho, já pode executar a aplicação `Relatorio Final Cocada.ipynb`.

## Dicas finais 

- Caso você queira adicionar algum outro método de criptografia, será necessário clonar o repositório FastText e obter os vetores das palavras criptografadas.

- Instruções adicionais podem ser encontradas na documentação deles: https://fasttext.cc/docs/en/crawl-vectors.html
