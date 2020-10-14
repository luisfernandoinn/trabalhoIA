# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: João Victor
"""
from sklearn import metrics
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN

iris = load_iris()
modelo = iris.target
dbscan = DBSCAN()
print(dbscan)
dbscan.fit(iris.data)
resultado = dbscan.labels_
print(pd.crosstab(modelo, resultado))
print(metrics.adjusted_mutual_info_score(modelo, resultado))