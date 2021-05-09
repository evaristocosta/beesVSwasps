# used to change filepaths
from sklearn.metrics import roc_curve, auc, accuracy_score, mean_squared_error
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from pipeline import load_bvsw


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score, plot_roc_curve


def escolhe_modelo(escolha):
    if escolha == '1':
        return SVC(kernel='linear', probability=True, random_state=42)
    elif escolha == '2':
        return GaussianNB()
    elif escolha == '3':
        return DecisionTreeClassifier(random_state=42)
    else:
        print("Opção não existe! Terminando...")
        exit()


def treino():
    print("Escolha o modelo:\n"
          "1 - SVM\n"
          "2 - Naive Bayes\n"
          "3 - Árvore de Decisão")
    escolha = input("Digite o valor: ")

    print("KFold ou Holdout repetido?\n"
          "1 - KFold\n"
          "2 - Holdout")
    validacao = input("Digite o valor: ")

    quantidade_dados = 1000  # máximo é 4000
    repeticoes = 10
    # para uso da validacao
    acertos = []
    aucs = []
    mses = []
    # pra plot de melhor roc
    falsos_positivos = []
    verdadeiros_positivos = []

    print("Separando dados, construindo modelo e gerando predições...")
    entrada, saida = load_bvsw(quantidade_dados)

    if validacao == '1':
        kf = KFold(n_splits=repeticoes, shuffle=True)

        for treino, teste in kf.split(entrada, saida):
            # redefine modelo a cada repeticao
            modelo = escolhe_modelo(escolha)

            # fit model
            modelo.fit(entrada[treino], saida[treino])

            # generate predictions
            predicao = modelo.predict(entrada[teste])

            # calculate accuracy
            acerto = accuracy_score(predicao, saida[teste])
            # predict probabilities for entrada[teste] using predict_proba
            probabilities = modelo.predict_proba(entrada[teste])
            # select the probabilities for label 1.0
            saida_proba = probabilities[:, 1]
            # calculate false positive rate and true positive rate at different thresholds
            false_positive_rate, true_positive_rate, _ = roc_curve(
                saida[teste], saida_proba, pos_label=1)

            # calculate AUC
            roc_auc = auc(false_positive_rate, true_positive_rate)
            # MSE
            mse = mean_squared_error(saida[teste], predicao)

            # 
            f1_score(saida[teste], predicao)
            recall_score(saida[teste], predicao)
            precision_score(saida[teste], predicao)
            accuracy_score(saida[teste], predicao)
            confusion_matrix(saida[teste], predicao)

            # guarda resultados
            acertos.append(acerto)
            aucs.append(roc_auc)
            mses.append(mse)
            falsos_positivos.append(false_positive_rate)
            verdadeiros_positivos.append(true_positive_rate)

    else:
        # divide em sets de treino e teste
        entrada_treino, entrada_teste, saida_treino, saida_teste = train_test_split(entrada,
                                                                                    saida,
                                                                                    test_size=.3,
                                                                                    random_state=1234123)

        # holdout repetido
        for _ in range(repeticoes):
            # redefine modelo a cada repeticao
            modelo = escolhe_modelo(escolha)

            # fit model
            modelo.fit(entrada_treino, saida_treino)

            # generate predictions
            predicao = modelo.predict(entrada_teste)

            # calculate accuracy
            acerto = accuracy_score(predicao, saida_teste)
            # predict probabilities for entrada_teste using predict_proba
            probabilities = modelo.predict_proba(entrada_teste)
            # select the probabilities for label 1.0
            saida_proba = probabilities[:, 1]
            # calculate false positive rate and true positive rate at different thresholds
            false_positive_rate, true_positive_rate, _ = roc_curve(
                saida_teste, saida_proba, pos_label=1)

            # calculate AUC
            roc_auc = auc(false_positive_rate, true_positive_rate)
            # MSE
            mse = mean_squared_error(saida_teste, predicao)

            # guarda resultados
            acertos.append(acerto)
            aucs.append(roc_auc)
            mses.append(mse)
            falsos_positivos.append(false_positive_rate)
            verdadeiros_positivos.append(true_positive_rate)

    print("-------------------------")
    print("Resultados:\n")

    print(f"Acerto médio: {np.average(acertos)} (+-{np.std(acertos)})")
    print(f"AUC médio: {np.average(aucs)} (+-{np.std(aucs)})")
    print(f"MSE médio: {np.average(mses)} (+-{np.std(mses)})")
    melhor_rodada = np.argmax(acertos)
    print("Melhor rodada: ", melhor_rodada)

    plt.title('Receiver Operating Characteristic')
    # plot the false positive rate on the x axis and the true positive rate on the y axis
    plt.plot(falsos_positivos[melhor_rodada],
             verdadeiros_positivos[melhor_rodada],
             label='AUC = {:0.2f}'.format(roc_auc))

    plt.legend(loc=0)
    plt.plot([0, 1], [0, 1], ls='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == "__main__":
    treino()
