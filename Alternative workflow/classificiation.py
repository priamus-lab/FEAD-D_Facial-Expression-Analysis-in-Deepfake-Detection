import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import cv2
from tensorflow import keras
import tensorflow as tf
import keras
import math
from keras.models import load_model,model_from_json
from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3
from random import randint
from random import sample 
import random
from keras.layers import Dense, Dropout, TimeDistributed
from tensorflow.keras.layers import LSTM, GRU, Bidirectional
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import time
import pickle5 as pickle
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"



def classification_model_fc(base_path):
  time_steps = 300
  features = 4096
  model = Sequential()
  model.add(Bidirectional(LSTM(units = 1024,return_sequences=True)))
  model.add(Bidirectional(LSTM(units = 512, return_sequences=True)))
  model.add(Bidirectional(LSTM(units = 512, return_sequences=False)))
  model.add(Dense(512, activation = 'relu'))
  model.add(Dense(256, activation = 'relu'))
  model.add(Dense(1, activation='sigmoid'))
  model.build((None, time_steps, features )) 
  model.summary()
  #print('----------------------------------Loading model form ' + base_path + '/best_model_LSTM_Iciap2023T.h5')
  model.load_weights(base_path + '/model_weights.h5')
  return model


 
def computePredictionFileList(all_video_files, classification_model, dataMainDirectory, outName):
  
  numeroDiVideo = len(all_video_files)
  i = 0                 
  video_start = 0
  n_rows = 300
  tot_prediction= pd.DataFrame()       
  print(f'Estraggo dal video {video_start} al {numeroDiVideo}')


  for nameVideo in all_video_files[video_start:numeroDiVideo+1]:   #numeroDiVideo+1 
    print('Working on ' + nameVideo)
    since = time.time()
    
    #with open(nameVideo, "rb") as fh:
    #  merged_df_fer67_IncV3 = pickle.load(fh)
    
    video_name = nameVideo.split('/')[-1]
    
    #video_name = merged_df_fer67_IncV3.index[0][0]
    merged_df_fer67_IncV3 = np.load(nameVideo)
    
    #merged_df_fer67_IncV3 = pd.read_csv(nameVideo)
    #x = [ element for element in merged_df_fer67_IncV3.columns if (element.startswith('f_I') or element.startswith('f_F'))]
    #merged_df_fer67_IncV3 = merged_df_fer67_IncV3[x]
    #merged_df_fer67_IncV3 = merged_df_fer67_IncV3.values
    
    if merged_df_fer67_IncV3.shape[0]>= n_rows:
      data_to_predict = merged_df_fer67_IncV3[0:n_rows,:]
    else:
      print('problema sotto i 300')
      #print(merged_df_fer67_IncV3.shape[0])
      data_to_predict = np.zeros((n_rows, merged_df_fer67_IncV3.shape[1]))
      
      dim_real = merged_df_fer67_IncV3.shape[0]
      reverse = merged_df_fer67_IncV3[::-1,:]
      data_to_predict[:dim_real,:] = merged_df_fer67_IncV3
      size_app = dim_real
      
      while  dim_real< n_rows:

        if n_rows> dim_real+size_app:
          data_to_predict[dim_real: (dim_real+size_app),:] = reverse
          reverse = reverse[::-1,:]
          dim_real = (dim_real+size_app)
          #print(dim_real)
        else:
          data_to_predict[dim_real:,:] = reverse[0: (n_rows - dim_real)]
          dim_real = n_rows

        
    p = classification_model.predict(data_to_predict[np.newaxis,:,:])
    print(f'          PREDICTION  {p} ')
    time_elapsed = time.time()-since 
    print('[TIME: %.0f m %.0f s]'  %(time_elapsed // 60, time_elapsed % 60))
    data = {'video_name': video_name, 'prediction': p[0,0], 'time': time_elapsed}
    tot_prediction = pd.concat((tot_prediction, pd.DataFrame( data = data, index=[0])), ignore_index=True)
    
  tot_prediction.to_csv(dataMainDirectory  + '/' + outName, index=False)
   

dataMainDirectory = '//' #PATH TO VIDEOS
classification_model = classification_model_fc('/') #PATH TO MODEL WEIGHTS

list_test = []  

#path to features extracted from videos 
files = os.listdir('//')
for element in files:
  list_test.append('//' + element)

print(len(list_test))

computePredictionFileList(list_test, classification_model, dataMainDirectory, 'predictionTEST.csv')