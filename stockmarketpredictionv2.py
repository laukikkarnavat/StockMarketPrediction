import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import csv
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Dropout,LSTM,BatchNormalization

#start = dt.datetime(2016,1,1)
#end = dt.datetime(2019,3,31)

#ALL CONSTANTS TO BE USED IN PROGRAM
SEQ_LEN=90
FUTURE_PERIOD_PREDICT=10
RATIO_TO_PREDICT="DLTR"

#result=web.DataReader('F','yahoo',startDay,endDay)

#!pwd

#result.to_csv('F.csv') #store in csv file

#pd.read_csv('F.csv') #read from csv to pandas dataframe

main_df=pd.DataFrame() #initialize empty dataframe

ratios=["BAC","F","GE","MSFT","DLTR"]
for ratio in ratios:
  dataset=f"{ratio}.csv"
  df=pd.read_csv(dataset)
  df.rename(columns={"High":f"{ratio}_High","Low":f"{ratio}_Low","Open":f"{ratio}_Open",
                    "Close":f"{ratio}_Close","Volume":f"{ratio}_Volume","Adj Close":f"{ratio}_Adj Close"},
           inplace=True)
  df.set_index("Date",inplace=True)
  
  if len(main_df) ==0:  
    main_df=df          #store df in main_df if main_df is empty   
  else:
    main_df=main_df.join(df) #concate all df dataframes to main_df

for c in main_df.columns:
  print(c)

main_df['future']=main_df[f"{RATIO_TO_PREDICT}_High"].shift(-FUTURE_PERIOD_PREDICT)
#create a future coloumn with value same as specific high coloumn which is shifted up by specific value

print(main_df[[f"{RATIO_TO_PREDICT}_High","future"]])

X=main_df.iloc[:,:-1] #Independent variable

for c in X.columns:
  print(c)

Y=main_df.iloc[:,30:31]  # Dependent coloum in Y

for c in Y.columns:
  print(c)

sc_X=StandardScaler()
sc_Y=StandardScaler()

X=sc_X.fit_transform(X)

Y=sc_Y.fit_transform(Y)

X[0][0]

Z=np.concatenate((X,Y),axis=1)

Z[0]

sequential_data=[]
prev_days=deque(maxlen=SEQ_LEN)

#Creates the complete sequential data to be split into train and test sequences
for i in range(0,len(Z)-1):
  prev_days.append([n for n in Z[i][:-1]])
  if len(prev_days) ==SEQ_LEN:
    sequential_data.append([np.array(prev_days),Z[i][-1]])

len(sequential_data[916][0])

sequential_data_train=[]
sequential_data_test=[]
sequential_data_train=sequential_data[0:908]
sequential_data_test=sequential_data[908:917]

type(sequential_data_train[0][0].tolist())

x_train=np.empty((908,90,30))
y_train=np.empty((908))

for i in range(len(sequential_data_train)):
  x_train[i]=sequential_data_train[i][0].tolist()

x_train.shape

for i in range(len(sequential_data_train)):
  y_train[i]=sequential_data_train[i][1]

y_train[0]

model=Sequential()

model.add(LSTM(128,input_shape=(90,30),return_sequences=True))  #value should be(90,30)
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128,input_shape=(90,30),return_sequences=True))  #value should be(90,30)
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128,input_shape=(90,30)))  #value should be(90,30)
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(1))

#optimizer
#opt=tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5)

model.compile(optimizer='rmsprop',
              loss='mse',
             metrics=['mae'])

model.fit(x_train,y_train,batch_size=10,epochs=1)

y_pred=sc_Y.inverse_transform(model.predict(x_train))

print(y_pred)

print(y_pred[0])

#create x_test data
x_test=np.empty((9,90,30))

for i in range(len(sequential_data_test)):
    x_test[i]=sequential_data_test[i][0].tolist()

print(x_test)

y_pred=sc_Y.inverse_transform(model.predict(x_test))

print(y_pred)

