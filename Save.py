# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 03:06:30 2018

@author: Helmi
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


df=pd.read_csv("sonar.all-data.csv")
df = shuffle(df)
#print(df.head())
#print(df.columns)
print(df.shape)
print(df.iloc[0][60])

training_data_X=df.iloc[0:190,0:60]
training_data_Y=df.iloc[0:190,60]

test_data_x=df.iloc[190:206,0:60]
test_data_y=df.iloc[190:206,60]
print(training_data_Y.head(20))
print(training_data_X.shape)
print(training_data_Y.shape)

lb_make = LabelEncoder()
training_data_Y = lb_make.fit_transform(training_data_Y)
print(training_data_Y)

model=Sequential()
model.add(Dense(15,activation='relu' ,input_dim=training_data_X.shape[1]))# first layer (only in the first layer we specify the shape of the input)
model.add(Dense(5,activation='relu'))
model.add(Dropout(0.5))#to avoid overfitting
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(training_data_X,training_data_Y,epochs=100,batch_size=10,verbose=True)

#Save model to disk
model_json=model.to_json()
with open ("model.json","w") as json_file:
    json_file.write(model_json)
#serialize weight to HDF5
model.save_weights("weight.h5")
print("saved model to disk")

#Evaluation
scores=model.evaluate(training_data_X,training_data_Y)
print(scores)

#predict and test
print(test_data_x.shape)
p=model.predict(test_data_x[0:20])
#print(taring[:15])
for i in range(16):
    #print(p[i])
    if(p[i]>0.5):
        print("Rock :)  Reality: "+test_data_y.iloc[i])
    else:
        print("Mine :o  Reality: "+test_data_y.iloc[i])