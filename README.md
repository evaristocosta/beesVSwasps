# BEES vs WASPS

Projeto de aprendizado de máquina que visa discriminar imagens de abelhas e vespas.

Base de dados (*dataset*): https://www.kaggle.com/jerzydziewierz/bee-vs-wasp

Para correto funcionamento:

1. Faça o download do *dataset*;
2. Crie uma pasta na raiz do projeto chamada `datasets`;
3. Dentro, coloque os arquivos baixados (o arquivo `labels.csv` e as pastas bee1, bee2, wasp1 e wasp2 são suficientes);
    - Em uma destas pastas há um arquivo entitulado `image.png`. Ele deve ser removido para correto funcionamento do programa (não é sabido o motivo de tal imagem estar nas pastas da base de dados);
4. Tenha `python3` e `pip` instalado na máquina;
5. Com um terminal aberto na pasta, rode o comando:
    ```bash
    pip install -r requirements.txt
    ```
6. Execute o programa com o comando:
    ```bash
    python classificador.py
    ```

## Notebook Jupyter

Há também disponível um arquivo em formato de notebook Jupyter que pode ser usado de maneira mais didática, para entender o funcionamento do programa.
