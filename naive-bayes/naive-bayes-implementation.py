#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: https://github.com/tigju
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
#sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})


class NaiveBayesClassifier():
    '''
    Teorema de Bayes 
    P(y|X) = P(X|y) * P(y) / P(X)
    '''
    def calc_prior(self, features, target):
        '''
        probabilidade a priori P(y)
        calculate prior probabilities
        '''
        self.prior = (features.groupby(target).apply(lambda x: len(x)) / self.rows).to_numpy()

        return self.prior
    
    def calc_statistics(self, features, target):
        '''
        calcula média(mean), varianca (var) para cada coluna e converte para numpy array
        ''' 
        self.mean = features.groupby(target).apply(np.mean).to_numpy()
        self.var = features.groupby(target).apply(np.var).to_numpy()
              
        return self.mean, self.var
    
    def gaussian_density(self, class_idx, x):     
        '''
        calculata a probabilidade da função de densidade gaussiana (distribuição normal)
        vamos assumir que a probabilidade de um valor alvo específico dada classe específica é normalmente distribuída 
        
        (1/√2pi*σ) * exp((-1/2)*((x-μ)^2)/(2*σ²)), quando +e a média, σ² é a variança, σ é a raíz da variança (desvio padrão)
        '''
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp((-1/2)*((x-mean)**2) / (2 * var))
#         numerator = np.exp(-((x-mean)**2 / (2 * var)))
        denominator = np.sqrt(2 * np.pi * var)
        prob = numerator / denominator
        return prob
    
    def calc_posterior(self, x):
        posteriors = []

        # calculata a probabilidade a posteriori de cada classe
        for i in range(self.count):
            prior = np.log(self.prior[i]) ## use o log para torná-lo mais estável numericamente
            conditional = np.sum(np.log(self.gaussian_density(i, x))) # use o log para torná-lo mais estável numericamente
            posterior = prior + conditional
            posteriors.append(posterior)
        # retorna a classe com a mai alta probabiliade a porteriori
        return self.classes[np.argmax(posteriors)]
     

    def fit(self, features, target):
        self.classes = np.unique(target)
        self.count = len(self.classes)
        self.feature_nums = features.shape[1]
        self.rows = features.shape[0]
        
        self.calc_statistics(features, target)
        self.calc_prior(features, target)
        
    def predict(self, features):
        preds = [self.calc_posterior(f) for f in features.to_numpy()]
        return preds

    def accuracy(self, y_test, y_pred):
        accuracy = np.sum(y_test == y_pred) / len(y_test)
        return accuracy

    def visualize(self, y_true, y_pred, target):
        
        tr = pd.DataFrame(data=y_true, columns=[target])
        pr = pd.DataFrame(data=y_pred, columns=[target])
        
        
        fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(15,6))
        
        sns.countplot(x=target, data=tr, ax=ax[0], palette='viridis', alpha=0.7, hue=target, dodge=False)
        sns.countplot(x=target, data=pr, ax=ax[1], palette='viridis', alpha=0.7, hue=target, dodge=False)
        

        fig.suptitle('Real vs Predicto - Comparação', fontsize=20)

        ax[0].tick_params(labelsize=12)
        ax[1].tick_params(labelsize=12)
        ax[0].set_title("Valores Reais", fontsize=18)
        ax[1].set_title("Valores Preditos", fontsize=18)
        plt.show()

if __name__ == '__main__':
  # upload Iris dataset -  shape is (150, 5)
  df = pd.read_csv("https://raw.githubusercontent.com/brunoreinaldo/ifrs-2021-ta/main/naive-bayes/data/water_potability.csv")
  # shuffle dataset with sample
  df = df.sample(frac=1, random_state=1).reset_index(drop=True)
  # df shape
  print(df.shape)
  
  # define previsores e classes
  X, y = df.iloc[:, :-1], df.iloc[:, -1]
  
  # # divide em treinamento e teste 0.7/0.3
  #X_train, X_test, y_train, y_test = X[:100], X[100:], y[:100], y[100:]
  X_train, X_test, y_train, y_test = X[:2292], X[2292:], y[:2292], y[2292:]

  print(X_train.shape, y_train.shape)
  print(X_test.shape, y_test.shape)

  x = NaiveBayesClassifier()
  x.fit(X_train, y_train)

  predictions = x.predict(X_test)
  print(x.accuracy(y_test, predictions))

  y_test.value_counts(normalize=True)

  x.visualize(y_test, predictions, 'Potability')