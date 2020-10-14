# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: João Victor
"""
from sklearn import metrics
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.cluster import DBSCAN

wine = load_wine()
modelo = wine.target
dbscan = DBSCAN(eps = 100, min_samples=50 )
print(dbscan)
dbscan.fit(wine.data)
resultado = dbscan.labels_
print(modelo)
print(resultado)
print(metrics.adjusted_mutual_info_score(modelo, resultado))