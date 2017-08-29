# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 21:43:45 2017

@author: Lenovo
"""

import pandas as pd
import numpy as np


data=pd.read_csv('Position_Salaries.csv')

data

X=data.iloc[:,1:2].values

Y=data.iloc[:,-1:].values

from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=300)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le=LabelEncoder()
X[:,0]=le.fit_transform(X[:,0])

ohe=OneHotEncoder(categorical_features=[0])
X=ohe.fit_transform(X).toarray()



from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=0)


model=reg.fit(X,Y)


model.predict(6.5)









