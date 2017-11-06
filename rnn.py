# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries

import matplotlib.pyplot as plt
#Using Quandl
import numpy as np
import pandas as pd
import quandl
import datetime



#importing the training set of AAPL

start = datetime.datetime(2013,1,1)
#
end=datetime.datetime(2017,11,5)

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
############################################
#get the LL HL LH HH

dataset['Diff_H']=dataset['Adj. High'].diff()
dataset['Diff_L']=dataset['Adj. Low'].diff()
dataset=dataset.dropna()

#dataset['Diff_L']=np.where(dataset['Diff_L']>0,'HL','LL')
#dataset['Diff_H']=np.where(dataset['Diff_H']>0,'HH','LH')



#dataset['status']=dataset['Diff_H']+dataset['Diff_L']
#del dataset['Diff_H']
#del dataset['Diff_L']

#original=dataset['status']

#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder = LabelEncoder()
##onehotencoder=OneHotEncoder(categorical_features=dataset['Diff_L'])
#dataset['Diff_L']=labelencoder.fit_transform(dataset['Diff_L'])
#
##onehotencoder.fit_transform(dataset['Diff_L']).toarray()
##Add the hot encoder



#convert DataFram to Array
dataset=dataset.values


## Feature Scaling
## Feature Scaling so we apply Normalization but not Standardisation
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#training_set_scaled = sc.fit_transform(dataset)
#
#
##try Robustscaler
#from sklearn.preprocessing import RobustScaler
#rc=RobustScaler()
#training_set_scaled=rc.fit_transform(dataset)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(dataset)


##try normalization
#from sklearn.preprocessing import Normalizer
#
#norm=Normalizer()
#training_set_normalized=norm.fit_transform(dataset)


#Creating a data strucure with 60 timesteps and one output
#n is the number of days back
n=60
X_train = []
y_train = []
for i in range(n,len(dataset)):
    X_train.append(training_set_scaled[i-n:i, :])
    y_train.append(training_set_scaled[i, 8:10])



X_train, y_train=np.array(X_train), np.array(y_train)




# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 10))

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and first LSTM layer
regressor.add(LSTM(units = 50, return_sequences=True, input_shape = (X_train.shape[1], 10)))
regressor.add(Dropout(0.2))

#adding second LSTM Layer
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))



# Adding last LSTM Layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#adding the output layer
regressor.add(Dense(units=2))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs =50, batch_size = 32)

# Part 3 - Making the predictions and visualising the results


inputs=dataset[len(dataset)-60:]
inputs = sc.transform(inputs)

#X_test=[]
#
#for i in range(1,2):
#    X_test.append(inputs[i-1:i,0])

#inputs=np.array(inputs)
X_test=np.array(inputs)
X_test = np.reshape(X_test, (1, X_test.shape[0], 10))


#X_test=np.reshape(inputs, (1,inputs.shape[0],8))


predicted_stock_price=regressor.predict(X_test)


new=np.zeros(shape=(len(X_test), 10) )

new[0,8:10]=predicted_stock_price

predicted_stock_price=sc.inverse_transform(new)


H_today=predicted_stock_price[0,8]+dataset[len(dataset)-1,7]
L_today=predicted_stock_price[0,9]+dataset[len(dataset)-1,6]


########
# Getting the predicted stock price of 2017
inputs = training_set[-60,:]
##inputs = sc.transform(inputs)
#inputs=np.reshape(inputs, (1, 1, 1))
#predicted_stock_price = regressor.predict(inputs)
#predicted_stock_price = sc.inverse_transform(predicted_stock_price)
######
#
#
#
#
## Getting the predicted stock price
##get all
#dataset_total=pd.concat((mystock_training['Adj. High'],mystock_test['Adj. High']),axis=0)
#inputs = dataset_total[len(dataset_total)-len(mystock_test)-60:].values
#inputs=inputs.reshape(-1,1)
#inputs=sc.fit_transform(inputs)

#X_test = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 6))

X_test = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 6))

predicted_stock_price = regressor.predict(inputs)
price=sc1.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()