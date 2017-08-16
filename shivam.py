#Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('DAta.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

#Adjusting the missing values

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Encoding categorical variables

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder()
le=le.fit(X[:,0])
X[:,0]=le.fit_transform(X[:,0])
ohe = OneHotEncoder(categorical_features=[0])
X=ohe.fit_transform(X).toarray()
le_y=LabelEncoder()
Y=le_y.fit_transform(Y)

#Split inti training and test data
from sklearn.cross_validation import train_test_split
train_X,test_X, train_Y, test_Y = train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
train_X=sc_x.fit_transform(train_X)
test_X=sc_x.transform(test_X)