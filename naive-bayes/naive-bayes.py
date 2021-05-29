# -*- coding: utf-8 -*-
"""
Created on Wed May 19 22:06:48 2021

@author: Bruno
"""

import pandas as pd
import numpy as np

base = pd.read_csv('https://github.com/rmanfredini/data/raw/main/risco_credito.csv')
#print(base)
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)
# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])
print(resultado)
print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)