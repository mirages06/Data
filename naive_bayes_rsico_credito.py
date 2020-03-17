# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 12:33:21 2019

@author: user
"""

import pandas as pd
import numpy as np

df = pd.read_csv('credit-data.csv')
df.loc[df.age <0, 'age'] = 40.92

X = df.iloc[:, 1:4].values
y = df.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:4])
X[:, 1:4] = imputer.transform(X[:,1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0 )

from sklearn.tree import DecisionTreeClassifier
classificador = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classificador.fit(X_train, y_train)
prevision = classificador.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(y_test, prevision)
matriz =confusion_matrix(y_test, prevision)