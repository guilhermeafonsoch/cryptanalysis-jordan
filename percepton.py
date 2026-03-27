# -*- coding: utf-8 -*-
"""
Perceptron de camada unica - problemas linearmente separaveis (OR gate)
"""

import numpy as np

# OR gate
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas = np.array([0, 1, 1, 1])

pesos = np.array([0.0, 0.0])
taxaDeAprendizagem = 0.1


def stepFunction(soma):
    return 1 if soma >= 1 else 0


def calculaSaida(registro):
    return stepFunction(registro.dot(pesos))


def treinar():
    global pesos
    epoca = 0
    while True:
        epoca += 1
        erroTotal = 0
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(entradas[i])
            erro = saidas[i] - saidaCalculada
            erroTotal += abs(erro)
            pesos += taxaDeAprendizagem * entradas[i] * erro

        if erroTotal == 0:
            print(f"Convergiu na epoca {epoca}")
            break


treinar()
print(f"\nPesos finais: {pesos}")
print("\nRede neural treinada:")
for i in range(len(entradas)):
    print(f"  {entradas[i]} -> {calculaSaida(entradas[i])}")
