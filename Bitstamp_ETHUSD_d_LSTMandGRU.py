import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame , concat
from sklearn.metrics import mean_absolute_error , mean_squared_error
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
from numpy import mean , concatenate
from math import sqrt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Activation,GRU
from numpy import array , hstack
from tensorflow import keras
import tensorflow as tf
import scipy.stats
import datetime


#********************************************************************************************************
# if you're using google drive to store the dataset
# from google.colab import drive
# drive.mount('/content/drive')
# url = '/yourDirectory/Dataset.csv'
url = "Poloniex_BTCUSDT_d.csv"
dataset = pd.read_csv(url)
x_1 = dataset['open'][1:] #open
x_2 = dataset['high'][1:] #high
x_3 = dataset['low'][1:] #low
x_4 = dataset['Volume BTC'][1:] #volume BTC
x_5 = dataset['Volume USDT'][1:] #volume USD
y = dataset['close'][1:] #Target

x_1_1 = x_1.values
x_2_1 = x_2.values
x_3_1 = x_3.values
x_4_1 = x_4.values
x_5_1 = x_5.values
y = y.values

    
    

#********************************************************************************************************
plt.figure(figsize=(100, 20))
plt.plot(x_1[:] , label='x1')
plt.plot(x_2[:] , label='x2')
plt.plot(x_3[:] , label='x3')
plt.plot(x_4[:] , label='x4')
plt.plot(x_5[:] , label='x5')
plt.plot(y[:] , label='y')
plt.legend(loc='upper right')
plt.title("Dataset" ,  fontsize=18)
plt.xlabel('Time step' ,  fontsize=18)
plt.ylabel('Values' , fontsize=18)
plt.legend()
plt.show()
#total of 28 batches of data
#Each variable in every batch contain 1258 --> 320 data points
#********************************************************************************************************
# Step 1 : convert to [rows, columns] structure
x_1 = x_1_1.reshape((len(x_1), 1)) #(9000, 1)
x_2 = x_2_1.reshape((len(x_2), 1)) #(9000, 1)
x_3 = x_3_1.reshape((len(x_3), 1)) #(9000, 1)
x_4 = x_4_1.reshape((len(x_4), 1)) #(9000, 1)
x_5 = x_5_1.reshape((len(x_5), 1)) #(9000, 1)
y = y.reshape((len(y), 1)) #(9000, 1)
print ("x_1.shape" , x_1.shape) 
print ("x_2.shape" , x_2.shape) 
print ("x_3.shape" , x_3.shape) 
print ("x_4.shape" , x_4.shape) 
print ("x_5.shape" , x_5.shape) 
print ("y.shape" , y.shape)
# Step 2 : normalization 
scaler = MinMaxScaler(feature_range=(0, 1))
x_1_scaled = scaler.fit_transform(x_1)
x_2_scaled = scaler.fit_transform(x_2)
x_3_scaled = scaler.fit_transform(x_3)
x_4_scaled = scaler.fit_transform(x_4)
x_5_scaled = scaler.fit_transform(x_5)
y_scaled = scaler.fit_transform(y)
# Step 3 : horizontally stack columns
dataset_stacked = hstack(( x_1_scaled,x_2_scaled,x_3_scaled,x_4_scaled,x_5_scaled, y_scaled))
print ("dataset_stacked.shape" , dataset_stacked.shape) #(9000, 3)
#********************************************************************************************************
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
 X, y = list(), list()
 for i in range(len(sequences)):
  #find the end of this pattern
  end_ix = i + n_steps_in
  out_end_ix = end_ix + n_steps_out-1
   #check if we are beyond the dataset
  if out_end_ix > len(sequences):
   break
   #raise Exception("out_end_ix > len(sequences)")
   #gather input and output parts of the pattern
  seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
  #seq_x(60,2) ; seq_y(30, )
  X.append(seq_x) #len(X)=8912; len(X[0])...len(X[8911])=60; len(X[0][0])...len(X[0][59]) =2
  y.append(seq_y) #len(y)=8912;len(y[0])...len(y[8911])=30;  
 return array(X), array(y)
# choose a number of time steps #change this accordingly
n_steps_in, n_steps_out = 5 , 3 #60, 30
n_features = 5
# covert into input/output
X, y = split_sequences(dataset_stacked, n_steps_in, n_steps_out)
print ("X.shape" , X.shape) #(8912, 60, 2)  ****************
print ("y.shape" , y.shape)  #(8912, 30)    ****************

split_point = int(len(X)*0.80) # 320*25
train_X , train_y = X[:split_point, :] , y[:split_point, :]
test_X , test_y = X[split_point:, :] , y[split_point:, :]

#********************************************************************************************************
#optimizer learning rate

bias_init = keras.initializers.Constant(value=0.2)
opt = keras.optimizers.Adam(learning_rate=0.0001)
# define model
model = Sequential()
#LSTM
# model.add(LSTM(80,activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features),bias_initializer=bias_init))
# model.add(LSTM(80,activation='relu',return_sequences=True))
# model.add(LSTM(80,activation='relu',return_sequences=True))
# model.add(LSTM(80,activation='relu',return_sequences=True))
# model.add(LSTM(80,activation='relu',return_sequences=True))
# model.add(LSTM(80, activation='relu',bias_initializer=bias_init))

#GRU
model.add(GRU(80,activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features),bias_initializer=bias_init))
# model.add(GRU(80,activation='relu',return_sequences=True))
# model.add(GRU(80,activation='relu',return_sequences=True))
# model.add(GRU(80,activation='relu',return_sequences=True))
# model.add(GRU(80,activation='relu',return_sequences=True))
model.add(GRU(80, activation='relu',bias_initializer=bias_init))

model.add(Dense(n_steps_out))
model.add(Activation('linear'))
model.summary()

model.compile(loss='mse' , optimizer=opt , metrics=['mse'])
timeStartTotal = time.time()
timeStart = time.localtime(timeStartTotal);
history = model.fit(train_X , train_y , epochs=300,batch_size=len(train_X),
                    verbose=1 , validation_data=(test_X, test_y))
#with tf.device('/device:GPU:0'):
    
# with tf.device('CPU'):
#     history = model.fit(train_X , train_y , epochs=2000,  
#                         verbose=1 , validation_data=(test_X, test_y) ,shuffle=False)
#model.fit ==> Find : epochs=20 , steps_per_epoch=60 , batch_size=64,
#                    verbose=1 , validation_data=(test_X, test_y) ,shuffle=False
timeEndTotal = time.time()
timeEnd = time.localtime(timeEndTotal);
total = int(timeEndTotal-timeStartTotal)
time_1 = datetime.timedelta(hours= timeStart.tm_hour , minutes=timeStart.tm_min, seconds=timeStart.tm_sec)

time_2 = datetime.timedelta(hours= timeEnd.tm_hour, minutes=timeEnd.tm_min, seconds=timeEnd.tm_sec)
estimate_time = time_2-time_1
print(f"Time used: {estimate_time}")
print(f"{total}")
#print(f"Start : {timeStartTotal/60}Min \nEnd : {timeEndTotal/60}Min \nTotal Use : {total}Min")
#********************************************************************************************************



#Find Metrics
def evaluate_predict(predictions, actual):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()

    return mse,rmse,mae
#Find_Correlation
oo = "1"
def correlation(predictions, actual):
    
    for i in range(n_steps_out):
        
        #********************************************************************************************************
        #Plot_1-3

        plt.figure(figsize=(9,3.5))
        plt.plot(predictions[:,[i]], color=(0.2,0.7,0.6,),lw=1.2, label='Predict')
        plt.plot(actual[:,[i]], 'r:', lw=2.5, label='Test Y')
        plt.title("Time_" + str(i+1) ,  fontsize=18)
        plt.xlabel('Data')
        plt.ylabel('Value')
        plt.legend(loc=2)
        plt.grid()
        #plt.savefig('C:/Users/rungs/Desktop/วิจัย/len(train_X)/bias0.2/IN5-OUT2/LSTM4/'+oo+'/Model'+ str(i+1)+'.png')
        #plt.savefig('C:/Users/rungs/Desktop/วิจัย/len(train_X)/bias0.2/IN5-OUT2/GRU4/'+oo+'/Model'+ str(i+1)+'.png')
        plt.show()
        
        #********************************************************************************************************
        # plt_correlation_1-3
        plt.figure(figsize=(9,3.5))
        plt.title('Correlation')
        plt.scatter(predictions[:,i], actual[:,i],alpha=0.5)
        axes = plt.gca()
        #plt.plot(np.unique(predictions[:,i]),
                #np.poly1d(np.polyfit(predictions[:,i],actual[:,i] ,1))(np.unique(predictions[:,i])), color='r',lw= 2.5)
        predic,act = np.polyfit(predictions[:,i],actual[:,i] ,1) 
        ploter = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
        plt.plot(ploter,predic*ploter+act,'-',color='r',lw=1.5)
        plt.xlabel('Predict')
        plt.ylabel('Actual')
        plt.title("Time_" + str(i+1) ,  fontsize=18)
        plt.grid()
        #plt.savefig('C:/Users/rungs/Desktop/วิจัย/len(train_X)/bias0.2/IN5-OUT2/LSTM4/'+oo+'/Cor'+ str(i+1)+'.png')
        #plt.savefig('C:/Users/rungs/Desktop/วิจัย/len(train_X)/bias0.2/IN5-OUT2/GRU4/'+oo+'/Cor'+ str(i+1)+'.png')
        plt.show()
        
        #********************************************************************************************************
        #Correlation_Result
        
        predictions1 = pd.Series(predictions[:,i])
        actual1 = pd.Series(actual[:,i])
        cor = predictions1.corr(actual1)
        print(cor)
        #print("Cor",str(i+1),cor)
    
    
    

y_pred_scaled = model.predict(test_X)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test = scaler.inverse_transform(test_y)
MSE, RMSE, MAE = evaluate_predict(y_pred, y_test)
corr = correlation(y_pred, y_test)
print(MSE)
print(RMSE)
print(MAE)
#print("MSE is ",MSE)
#print("RMSE is ", RMSE)
#print("MAE is ", MAE)
