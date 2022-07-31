import sys,os
os.system('cls')

from unicodedata import category
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline 
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10 
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from  sklearn.preprocessing import MinMaxScaler

my_columns = ['OrderDate', 'Sales']

df = pd.read_csv('Sample-Superstore.csv',  usecols=my_columns , nrows=300 , sep=',', encoding='cp1252', engine='python')

### Data Manipulation ###

df=df.replace({'\$':''},regex=True)

df = df.astype({"Sales": float})
df["OrderDate"] = pd.to_datetime(df.OrderDate, format="%m/%d/%Y")
df.dtypes

df.index=df['OrderDate']

### Data Visualization ###

#plt.plot(df['Sales'],label='Sales Price history')
#plt.plot(df['OrderDate'],df['Sales'])
#df.plot(x='OrderDate', y ='Sales',  kind = 'line')	

#plt.show()

### LSTM Prediction Model ###
df = df.sort_index(ascending=True,axis=0)

data = pd.DataFrame(index=df.index,columns=['OrderDate','Sales'])

for i in range(0,len(data)):
    data["OrderDate"][i]=df['OrderDate'][i]
    data["Sales"][i]=df["Sales"][i]


### Min-Max Scaler ###

scaler=MinMaxScaler(feature_range=(0,1))

data.index=data.OrderDate
data.drop('OrderDate' ,axis=1, inplace=True)

final_data = data.values
train_data=final_data[0:200,:]
valid_data=final_data[200:,:]

scaler=MinMaxScaler(feature_range=(0,1))

scaled_data=scaler.fit_transform(final_data)
x_train_data,y_train_data=[],[]
for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])

x_train_data = np.asarray(x_train_data)
y_train_data = np.asarray(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1],1))

### LSTM Model ###

lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(np.shape(x_train_data)[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

model_data=data[len(data)-len(valid_data)-60:].values
model_data=model_data.reshape(-1,1)
model_data=scaler.transform(model_data)

### Train and Test Data ###

lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

X_test=[]
for i in range(60,model_data.shape[0]):
    X_test.append(model_data[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

### Prediction Function ###

predicted_stock_price=lstm_model.predict(X_test)
predicted_stock_price=scaler.inverse_transform(predicted_stock_price)

### Prediction Result ###

train_data=data[:200]
valid_data=data[200:]
valid_data['Predictions']=predicted_stock_price
plt.plot(train_data['Sales'])
plt.plot(valid_data[['Sales','Predictions']])

plt.show()

### Export Data to csv file for Comparing in Tableau ###

df2=train_data.merge(valid_data, how='outer',  on='OrderDate')

df2.to_csv('comparing.csv', index=True)

    

