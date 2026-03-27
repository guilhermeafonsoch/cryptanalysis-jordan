# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 00:43:50 2021

@author: guilh
"""

from sklearn.neural_network import MLPClassifier
from sklearn import datasets

data_iris = datasets.load_iris()

entradas= data_iris.data
saidas = data_iris.target 

rede_neural = MLPClassifier(verbose=True, 
                            max_iter=1000,
                            tol = 0.00001,
                            activation = "logistic",
                            learning_rate_init = 0.001,
                            )

rede_neural.fit(entradas, saidas)

resultado = rede_neural.predict([[5, 7.2, 5.1, 10]])
classes = data_iris.target_names
print(f"\nPredicao para [5, 7.2, 5.1, 10]: {classes[resultado[0]]} (classe {resultado[0]})")
print(f"Acuracia no treino: {rede_neural.score(entradas, saidas):.4f}")
