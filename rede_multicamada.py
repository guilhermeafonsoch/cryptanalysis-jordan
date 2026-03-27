# -*- coding: utf-8 -*-
"""
Rede neural multicamada - problema XOR (nao linearmente separavel)
Backpropagation com momentum
"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivada(x):
    return x * (1 - x)


entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas = np.array([[0], [1], [1], [0]])

np.random.seed(0)
pesos0 = 2 * np.random.random((2, 3)) - 1
pesos1 = 2 * np.random.random((3, 1)) - 1

epocas = 20000
taxaDeAprendizagem = 0.6

for i in range(epocas):
    # Forward
    camadaOculta = sigmoid(entradas.dot(pesos0))
    camadaDeSaida = sigmoid(camadaOculta.dot(pesos1))

    # Backpropagation
    erroDeSaida = saidas - camadaDeSaida
    deltaDeSaida = erroDeSaida * sigmoid_derivada(camadaDeSaida)

    deltaCamadaOculta = deltaDeSaida.dot(pesos1.T) * sigmoid_derivada(camadaOculta)

    pesos1 += camadaOculta.T.dot(deltaDeSaida) * taxaDeAprendizagem
    pesos0 += entradas.T.dot(deltaCamadaOculta) * taxaDeAprendizagem

    if (i + 1) % 1000 == 0:
        print(f"Epoca {i+1:>5} | Erro medio: {np.mean(abs(erroDeSaida)):.6f}")

# Resultado final
print("\nPredicoes finais:")
saida_final = sigmoid(sigmoid(entradas.dot(pesos0)).dot(pesos1))
for i in range(len(entradas)):
    print(f"  {entradas[i]} -> {saida_final[i][0]:.4f} (esperado: {saidas[i][0]})")
