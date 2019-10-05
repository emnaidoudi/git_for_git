# -*- coding: utf-8 -*-
"""
Created on Sun May 27 05:08:16 2018

@author: Helmi
"""
import pandas as pd
from sklearn.utils import shuffle
from keras.models import model_from_json
import datetime as dt
from flask import Flask
from flask import request
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

def clear_data(df):
    df=df.drop('Name',axis=1) # we don't need Name column since it doesn't contribute in the result
    lb_make = LabelEncoder()
    df['Sex'] = lb_make.fit_transform(df['Sex']) # male =>1, female =>0
    df['Sex'] = lb_make.fit_transform(df['Sex']) # male =>1, female =>0
    df=df.dropna()
    df['Embarked']=lb_make.fit_transform(df['Embarked']) # S Q C //be careful there's null values => df.dropna(à)
    df['Cabin']=lb_make.fit_transform(df['Cabin'])
    df['Ticket']=lb_make.fit_transform(df['Ticket'])
    return df
    
    
df =pd.read_csv('train_titanic.csv')
df=clear_data(df)


taining_data=df.loc[:,df.columns != 'Survived']
survived_or_not=df.loc[:,df.columns == 'Survived']

model=Sequential()
model.add(Dense(taining_data.shape[1],activation='relu' ,input_dim=taining_data.shape[1]))# first layer (only in the first layer we specify the shape of the input)
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.5))#to avoid overfitting
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_logarithmic_error',optimizer='adam')
model.fit(taining_data,survived_or_not,epochs=10,batch_size=10,verbose=True)


#Save model to disk
model_json=model.to_json()
with open ("model_titanic.json","w") as json_file:
    json_file.write(model_json)
#serialize weight to HDF5
model.save_weights("weight_titanic.h5")
print("saved model to disk")