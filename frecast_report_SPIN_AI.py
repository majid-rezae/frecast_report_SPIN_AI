import sys
 
 
 
from Extensions.TKDataAccess import TKDataAccess



def new () :   
    ####import all the packages####
    import pymongo
    import json 
    from pandas import read_csv
    from pandas import to_datetime
    from datetime import datetime
    from pandas import DataFrame
    import math
    import string
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    from keras.models import Sequential
    from keras.layers import  Dense,LSTM,Dropout
    from tensorflow.keras import backend  
    from tensorflow.keras.models import Sequential
    from pandas.tseries.offsets import DateOffset
    from datetime import datetime,date
    
    ####import and read database csv file####
    dev_eui='8cf9574000000012'
    
    myclient = pymongo.MongoClient("mongodb://ibti:ibti@iotibti.ddns.net:27017/admin?tls=true")
    mydb = myclient["data"]
    #dev_eui='8cf9574000000012'
    col_data = mydb[dev_eui]

    lista_dados = []

    lista_tempo= []
    tempo = 0

    for item in col_data.find():
        if int(item['ts'])- tempo >= 3600*24:
            tempo=int(item['ts'])
            dado=float(item['temp'])
            lista_tempo.append(datetime.utcfromtimestamp(tempo).strftime('%Y-%m-%d'))
            lista_dados.append(dado)
            
    lista_tempo.reverse()
    lista_dados.reverse()

    del lista_tempo[100:]
    del lista_dados[100:]

    lista_tempo.reverse()
    lista_dados.reverse()

    dic={'ds':lista_tempo, 'y':lista_dados}
    df=pd.DataFrame(dic) 

    ####select Data column as index####
    df["ds"] =pd.to_datetime(df.ds)
    df=df.set_index ('ds')
    #dataset=dataset.sort_values(by='Data')

    ####filter a select column####
    df= df.replace(',','.', regex=True)
    #df = dataset.filter(["Velocidade do vento (m/s)"])
    #print(df)



    # set datas between 0 and 1 for neural network model  
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(df)
    X_replace= df.replace(',','.', regex=True)
    df_scaled = scaler.fit_transform(df)
    # convert it back to numpy array
    X_np = X_replace.values
    # set the object type as float
    X_fa = X_np.astype(float)
    # perdict for seven days
    forecast_features_set = []
    labels = []
    for i in range(7,len(df)):
        forecast_features_set.append(df_scaled[i-7:i, 0])
        labels.append(df_scaled[i, 0])


        
    forecast_features_set , labels = np.array(forecast_features_set ), np.array(labels)

    forecast_features_set = np.reshape(forecast_features_set, (forecast_features_set.shape[0], forecast_features_set.shape[1], 1))
    forecast_features_set.shape

    # LSTM Model 
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(forecast_features_set.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100))
    model.add(Dropout(0.2))

    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    history = model.fit(forecast_features_set, labels, epochs = 1, batch_size = 100,verbose=0)

    forecast_list=[]

    batch=df_scaled[-forecast_features_set.shape[1]:].reshape((1,forecast_features_set.shape[1],1))

    for i in range(forecast_features_set.shape[1]):
        forecast_list.append(model.predict(batch)[0])
        batch = np.append(batch[:,1:,:], [[forecast_list[i]]], axis=1)
    df_predict=pd.DataFrame(scaler.inverse_transform(forecast_list),index=df[-forecast_features_set.shape[1]:].index,columns=["forecasting"])  
                                

    df_predict =pd.concat([df,df_predict],axis=1)


    add_dates=[df.index[-1]+DateOffset(days=x) for x in range(0,8)]
    future_dates=pd.DataFrame(index=add_dates[1:],columns=df.columns)
    #df_forecast=pd.DataFrame(scaler.inverse_transform(forecast_list),index=future_dates[-forecast_features_set.shape[1]:].index) 
                            
    df_forecast=pd.DataFrame(scaler.inverse_transform(forecast_list),index=future_dates[-forecast_features_set.shape[1]:].index, 
                            columns=["forecasting"])                           
    df_forecast =pd.concat([df,df_forecast],axis=1)
    df_forecast=df_forecast.drop(['y'], axis=1)
    df_forecast=df_forecast.dropna()
        
    df_forecast=df_forecast.reset_index()
    df_forecast.index.name = 'foo' 
    df_forecast.index = ['Row_1', 'Row_2', 'Row_3', 'Row_4', 'Row_5', 'Row_6', 'Row_7']
    df_forecast['index'] = pd.to_datetime(df_forecast['index']).dt.date
    df_forecast['forecasting']=  df_forecast['forecasting'].apply('{:,.4f}'.format)
        
    #df_forecast_new.index.name = 'foo'               

    #### save data to CSV file ###s#
    #df_forecast.to_csv(r'Forcasting_Velocidade.csv', index = True, header=True)
    #df_forecast = df_forecast.set_axis(['dia', 'forecasting'], axis=1, inplace=False)
    


    lista_return=[]
    lista_return.append(df_forecast['forecasting']['Row_1'])
        
        #print(lista_return) 
        
    
    day1= df_forecast ['index']['Row_1']  
    day2= df_forecast ['index']['Row_2']  
    day3= df_forecast ['index']['Row_3']  
    day4= df_forecast ['index']['Row_4']  
    day5= df_forecast ['index']['Row_5']  
    day6= df_forecast ['index']['Row_6']  
    day7= df_forecast ['index']['Row_7']  
    fore1=df_forecast ['forecasting']['Row_1'] 
    fore2=df_forecast ['forecasting']['Row_2'] 
    fore3=df_forecast ['forecasting']['Row_3'] 
    fore4=df_forecast ['forecasting']['Row_4'] 
    fore5=df_forecast ['forecasting']['Row_5'] 
    fore6=df_forecast ['forecasting']['Row_6'] 
    fore7=df_forecast ['forecasting']['Row_7'] 

    return  day1,day2,day3,day4,day5,day6,day7,fore1,fore2,fore3,fore4,fore5,fore6,fore7
aa=new() 



dataAccess = TKDataAccess()
connectionStatus = dataAccess.Connect("127.0.0.1:3101", "guest", "")
print("Connection: " + connectionStatus)

forecast1=float(aa[7] )
forecast2=float(aa[8] )
forecast3=float(aa[9] )
forecast4=float(aa[10] )
forecast5=float(aa[11] )
forecast6=float(aa[12] )
forecast7=float(aa[13] )
 

data0=aa[0] 
dt0=data0.strftime('%Y%m%d')
yy0=int(str(dt0)[0:4])
mm0=int(str(dt0)[4:6])
dd0=int(str(dt0)[6:8])  

data1=aa[1] 
dt1=data1.strftime('%Y%m%d')
yy1=int(str(dt1)[0:4])
mm1=int(str(dt1)[4:6])
dd1=int(str(dt1)[6:8]) 

data2=aa[2] 
dt2=data2.strftime('%Y%m%d')
yy2=int(str(dt2)[0:4])
mm2=int(str(dt2)[4:6])
dd2=int(str(dt2)[6:8]) 

data3=aa[3] 
dt3=data3.strftime('%Y%m%d')
yy3=int(str(dt3)[0:4])
mm3=int(str(dt3)[4:6])
dd3=int(str(dt3)[6:8]) 

data4=aa[4] 
dt4=data4.strftime('%Y%m%d')
yy4=int(str(dt4)[0:4])
mm4=int(str(dt4)[4:6])
dd4=int(str(dt4)[6:8]) 

data5=aa[5] 
dt5=data5.strftime('%Y%m%d')
yy5=int(str(dt5)[0:4])
mm5=int(str(dt5)[4:6])
dd5=int(str(dt5)[6:8]) 

data6=aa[6] 
dt6=data6.strftime('%Y%m%d')
yy6=int(str(dt6)[0:4])
mm6=int(str(dt6)[4:6])
dd6=int(str(dt6)[6:8])

dataAccess.SetObjectValue("Tag.yymmout000", mm0);  
if mm0 == 10 :
  mm10='October' 
elif mm0 == 11  :
  mm10='September'   
elif mm0 == 12 :
  mm10='December'  
dataAccess.SetObjectValue("Tag.yymmout", mm10);  

dataAccess.SetObjectValue("Tag.yymmout111", mm1);
if mm1 == 10 :
  mm11='October'
elif mm1 == 11  :
  mm11='September'   
elif mm1 == 12 :
  mm11='December' 
dataAccess.SetObjectValue("Tag.yymmout1", mm11);

dataAccess.SetObjectValue("Tag.yymmout222", mm2);
if mm2 == 10 :
  mm12='October' 
elif mm2 == 11  :
  mm12='September'  
elif mm2 == 12 :
  mm12='December' 
dataAccess.SetObjectValue("Tag.yymmout2", mm12);

dataAccess.SetObjectValue("Tag.yymmout333", mm3);
if mm3 == 10 :
  mm13='October' 
elif mm3 == 11  :
  mm13='September'   
elif mm3 == 12 :
  mm13='December' 
dataAccess.SetObjectValue("Tag.yymmout3", mm13);

dataAccess.SetObjectValue("Tag.yymmout444", mm4);
if mm4 == 10 :
  mm14='October' 
elif mm4 == 11  :
  mm14='September'   
elif mm4 == 12 :
  mm14='December' 
dataAccess.SetObjectValue("Tag.yymmout4", mm14);  

dataAccess.SetObjectValue("Tag.yymmout555", mm5);
if mm5 == 10 :
  mm15='October' 
elif mm5 == 11  :
  mm15='September'   
elif mm5 == 12 :
  mm15='December'  
dataAccess.SetObjectValue("Tag.yymmout5", mm15);   

dataAccess.SetObjectValue("Tag.yymmout666", mm6);
if mm6 == 10 :
  mm16='October' 
elif mm6 == 11  :
  mm16='September'   
elif mm6 == 12 :
  mm16='December' 
dataAccess.SetObjectValue("Tag.yymmout6", mm16);  

 
 
 
 
 
 
 
dataAccess.SetObjectValue("Tag.yy00", yy0);
dataAccess.SetObjectValue("Tag.yydd0", dd0);
dataAccess.SetObjectValue("Tag.yydd2", dd2); 
dataAccess.SetObjectValue("Tag.yyy2", yy2);
dataAccess.SetObjectValue("Tag.yyy1", yy1);
dataAccess.SetObjectValue("Tag.yydd1", dd1);
dataAccess.SetObjectValue("Tag.yydd3", dd3); 
dataAccess.SetObjectValue("Tag.yyy3", yy3);
dataAccess.SetObjectValue("Tag.yydd4", dd4); 
dataAccess.SetObjectValue("Tag.yyy4", yy4);
dataAccess.SetObjectValue("Tag.yydd5", dd5); 
dataAccess.SetObjectValue("Tag.yyy5", yy5);
dataAccess.SetObjectValue("Tag.yydd6", dd6); 
dataAccess.SetObjectValue("Tag.yyy6", yy6);
dataAccess.SetObjectValue("Tag.yyfore1", forecast1);
dataAccess.SetObjectValue("Tag.yyfore2", forecast2);
dataAccess.SetObjectValue("Tag.yyfore3", forecast3);
dataAccess.SetObjectValue("Tag.yyfore4", forecast4);
dataAccess.SetObjectValue("Tag.yyfore5", forecast5);
dataAccess.SetObjectValue("Tag.yyfore6", forecast6);
dataAccess.SetObjectValue("Tag.yyfore7", forecast7);
 