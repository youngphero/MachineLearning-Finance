# XGBoost

# Install xgboost following the instructions on this link: http://xgboost.readthedocs.io/en/latest/build.html#

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
#Using Quandl

import quandl
import datetime



#importing the training set of AAPL

start = datetime.datetime(2013,1,1)
#
end=datetime.datetime(2017,11,2)

#start_test=datetime.datetime(2017,10,21)
#end_test=datetime.datetime(2017,10,3)

#Getting HH HL LL LH of a stock
mystock_training=quandl.get('WIKI/AAPL',start_date=start,end_date=end)



#keeping only the heigh and low with volume
mystock_training=mystock_training[['Adj. Open','Adj. Close','Adj. Volume','Adj. Low','Adj. High']]

#mystock_training['Date']=mystock_training.index

#getting the sentimental dada
mysent_training=quandl.get('NS1/AAPL_CI',api_key='No8oze7d5V48kqY_rSuy',start_date=start,end_date=end)
mysent_training=mysent_training[['Sentiment','News Volume','News Buzz']]
#mysent_training['Date']=mysent_training.index

training_set=mysent_training.join(mystock_training)
dataset=training_set.dropna()
# Importing the dataset

X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 7:8].values

# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X_1 = LabelEncoder()
#X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#labelencoder_X_2 = LabelEncoder()
#X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training set

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()