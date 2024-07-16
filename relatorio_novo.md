# Temática do Projeto: Comparação da Eficiência dos Métodos de Criptografia por meio do PCA.

## Motivação

- Desejo avaliar a eficiência de diferentes métodos de Criptografia por meio da utilização de técnicas de SVD (Singular Value Decomposition) e PCA (Principal Component Analysis).

- Portanto, vou aplicar técnicas de criptografias variadas a poemas, textos famosos da literatura, 

- Dados diferentes métodos de criptografia, desejo avaliar a eficiência de cada método comparativamente, para isso, vou considerar 

- Irei me concentrar nas seguintes técnicas de criptografia para essa análise:

    - Substituição simples (Cifra de César)

    - Cifra Multiplicativa

    - RSA

## Principais Dificuldades

1. Decomposição em Componentes Principais de palavras fora do vocabulário (Criptografadas)

    - A presença de palavras fora do vocabulário, também conhecidas como OOV (Out of Vocabulary Words), 

- Primeiramente, precisamos considerar o problema de obter a decomposição em componenets principais para os textos criptografados. Como sabemos, o processo de conversão de texto para vetor, conhecido como Word Embedding, corresponde ao 

## Modelagem e Recursos

### Amostras

Para facilitar as análises, vamos antes tratar as de palavras, isto é, remover as pontuações e os acentos presentes

Poema de Vinícius de Moraes:

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

- Versão criptografada com Cifra de César sujeita a deslocamento em 3 unidades + tratada:

    Gh wxgr dr phx dpru vhul dwhqwr
    Dqwhv h frp wdo chr h vhpuh h wdqwr
    Txh phvpr hp idfh gr pdlru hqfdqwr
    Ghof vh hqfdqwh plhv phx shqvdphqwr

    Txhur ylvelr hp fdgd vao prphqwr
    H hp orxyrxk klm gh hvsdokdu phx fdqwr
    H ulu phx ulvr h ghhuududr phx sudqwr
    Dr vhx shvdu rx vhx frqwhqwdphqwr

    H dvvlp txdqgr pllv wdugh ph surfxuh
    Txhp ndeg d pruwh dqjvwl d txhp ylwh
    Txhp ndeg d solidar ilp gh txhp dpd

    Hx srvvd ph glchru gr dpru txh wlyh
    Txh qdr vhma lprwdor srvwr txh e fkdpd
    Pdv txh vhma lqilqlwr hqtxdqw dxfh


Cifra de César (exemplo clássico)

# Relação com Decomposição em Componentes Principais

# Análise em si e Clusterização

Após a disposição dos dados 

# Critérios de Similaridade

- Proximidade das palarvas na no gráfico modelado por 3 PCAs.

- Palavras contidas no mesmo Cluster.

# Referências

- https://fasttext.cc/docs/en/unsupervised-tutorial.html

- 
