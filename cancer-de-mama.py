# -*- coding: utf-8 -*-
"""
Classificacao binaria - Cancer de mama (dataset UCI)
Rede neural multicamada com backpropagation
"""

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivada(x):
    return x * (1 - x)


# Dataset
base = datasets.load_breast_cancer()
entradas = StandardScaler().fit_transform(base.data)
saidas = base.target.reshape(-1, 1)

np.random.seed(42)
pesos0 = np.random.randn(30, 10) * 0.1
pesos1 = np.random.randn(10, 1) * 0.1

epocas = 5000
taxaDeAprendizagem = 0.01

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

    if (i + 1) % 500 == 0:
        mae = np.mean(abs(erroDeSaida))
        acc = np.mean((camadaDeSaida > 0.5).astype(int) == saidas)
        print(f"Epoca {i+1:>5} | Erro medio: {mae:.6f} | Acuracia: {acc:.4f}")

# Resultado final
predicoes = (camadaDeSaida > 0.5).astype(int)
acuracia = np.mean(predicoes == saidas)
print(f"\nAcuracia final: {acuracia:.4f} ({int(acuracia * len(saidas))}/{len(saidas)})")
