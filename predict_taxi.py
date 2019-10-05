# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 04:15:57 2018

@author: Helmi
"""
import pandas as pd
from sklearn.utils import shuffle
from keras.models import model_from_json
import datetime as dt
from flask import Flask
from flask import request
from sklearn.preprocessing import LabelEncoder


import ast

app=Flask(__name__)

test_taxi=pd.read_csv("test.csv")

def isWeekend(el):
    if dt.date(int(el['year']), int(el['month']), int(el['day'])).weekday() > 5:
        return 1
    else:
        return 0

def dis_central(el):
    return abs(el['pickup_longitude']+73.968285)+abs(el['pickup_latitude']-40.785091)

def dis_manha(el):
   return abs(el['pickup_longitude']+73.985130)+abs(el['pickup_latitude']- 40.758896)

def dis_brock(el):
   return abs(el['pickup_longitude']+73.949997)+abs(el['pickup_latitude']-40.650002)

def transform_date(df):
    df[['date','time']]=df['pickup_datetime'].str.split(' ',expand=True)
    df[['year','month','day']]=df['date'].str.split('-',expand=True)
    df[['minute','x','y']]=df['time'].str.split(':',expand=True)
    df=df.drop(['date','time','pickup_datetime','x','y'],1)
    df["weekend"] = df.apply(isWeekend, axis=1)
    df['dist_central']=df.apply(dis_central,axis=1)
    df['dist_manha']=df.apply(dis_manha,axis=1) 
    df['dist_brock']=df.apply(dis_brock,axis=1) 
    df = df.drop('pickup_longitude', 1)
    df = df.drop('pickup_latitude', 1)
    lb_make = LabelEncoder()
    df['store_and_fwd_flag'] = lb_make.fit_transform(df['store_and_fwd_flag'])
    df=df.drop('id',1)

    return df

test_taxi=transform_date(test_taxi)
print(test_taxi.shape)
print(test_taxi.columns)
#load json and create model
json_file=open('model_taxi.json','r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
    
loaded_model.load_weights("weight_taxi.h5")#  we use the same data ( we don't fit new data)
print("Loaded model from disk")

p=loaded_model.predict(test_taxi)
for i in range(16,40):
    print('**',p[i])
    
    
  
@app.route('/predict')
def to_predict():
    #age=request.args.get("age")
    
      #load json and create model
    json_file=open('model_taxi.json','r')
    loaded_model_json=json_file.read()
    json_file.close()
    loaded_model=model_from_json(loaded_model_json)
    
    loaded_model.load_weights("weight_taxi.h5")#  we use the same data ( we don't fit new data)
    print("Loaded model from disk")
    v_id =request.args.get("1")
    pass_count= request.args.get("2")
    pick_up =request.args.get("3")
    drop_long=request.args.get("4")
    drop_lat=request.args.get("5")
    store=request.args.get("6")
      
    df= pd.DataFrame(data=[[id,v_id,pick_up,pass_count,pick_long,pick_lat,drop_lat,drop_long,store]],
                     columns=test_taxi.columns
            )
    print(df)
    df = transform_date(df)
    
    return loaded_model.predict(df)
      
if __name__=='__main__':
    app.run() 