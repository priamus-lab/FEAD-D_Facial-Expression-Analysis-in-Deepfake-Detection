import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

'''
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
'''

def importModels(dataMainDirectory):
    print('LOADING MODELS...')    
    print('------------------------------------ LOADING fer67AccMicroExpreDetector')
    json_file = open(f'{dataMainDirectory}/fer67AccMicroExpreDetector.json', 'r') #/m100_work/IscrC_FEAD-D/utils_FacePreprocessing/fer67AccMicroExpreDetector.json
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f'{dataMainDirectory}/fer67AccMicroExpreDetector.h5') #/m100_work/IscrC_FEAD-D/utils_FacePreprocessing/fer67AccMicroExpreDetector.h5
    from keras.models import Sequential
    fer67_cnn = Sequential()
    for layer in loaded_model.layers[:-7]: # just exclude last layer from copying
        fer67_cnn.add(layer)
    print('------------------------------------ LOADING InceptV3')
    incV3_cnn = InceptionV3(include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling='max',
    )

    return fer67_cnn, incV3_cnn
    
    
      
def features_extraction_step2(x_data_48, x_data_299, fer67_cnn, incV3_cnn ):
  x_data_48 = np.expand_dims(x_data_48, axis=-1)
  predictions2048_48 = fer67_cnn.predict(x_data_48, batch_size = 16)
  print(predictions2048_48.shape)
  #print('check nan su predictions2048_48')
  if(np.isnan(predictions2048_48).any()):
    print('----------------- PROBLEMA SUI 48')
  #print('CREO DataFrame - x_data - videoName - frameID')
  columns_Fer67_2048 =[('f_Fer' + str(i)) for i in range(2048)] #f1,f2,f3....fn
  df_fer67_2048Features = pd.DataFrame(predictions2048_48,columns = columns_Fer67_2048)
  #df_fer67_2048Features['video'] = ofVideo
  df_fer67_2048Features['frame_id'] = np.arange(df_fer67_2048Features.shape[0]).tolist()
  df_fer67_2048Features = df_fer67_2048Features.set_index('frame_id')

  x_data_preproc_299 = keras.applications.inception_v3.preprocess_input(x_data_299)
  #print('check nan x_data_preproc_299')
  if (np.isnan(x_data_preproc_299).any()):
    print('----------------- PROBLEMA SUI 299')
  predictions_299 = incV3_cnn.predict(x_data_preproc_299, batch_size = 16)
  if (np.isnan(predictions_299).any()):
    print('----------------- PROBLEMA SUI 299 v2')
  print(predictions_299.shape)
  #print('check nan predictions_299')
  #print(np.isnan(predictions_299).any())
  # print('CREO DataFrame - x_data - videoName - frameID')
  columns_InceptionV3 =[('f_Inc' + str(i)) for i in range(2048)] #f1,f2,f3....fn
  df_incV3_2048Features = pd.DataFrame(predictions_299,columns = columns_InceptionV3)
  df_incV3_2048Features['frame_id'] = np.arange(df_incV3_2048Features.shape[0]).tolist()
  df_incV3_2048Features = df_incV3_2048Features.set_index('frame_id')
  merged_df_fer67_IncV3 = pd.merge(df_fer67_2048Features, df_incV3_2048Features, left_index=True, right_index=True, how='inner',validate="one_to_one")
  #merged_df_fer67_IncV3 = pd.merge(df_fer67_2048Features, df_incV3_2048Features, left_on = 'frame_id', right_on = 'frame_id', how='inner',validate="one_to_one")
  print(merged_df_fer67_IncV3.shape)
  return merged_df_fer67_IncV3
  



parser = argparse.ArgumentParser()
parser.add_argument('--initial', type = int, default=1, help='Videos directory')
parser.add_argument('--final', type = int, default=1, help='Videos directory')
opt = parser.parse_args()
print(opt)

#test_df = pd.DataFrame(pd.read_csv(dataMainDirectory + 'Our_Approach/labels_TestSet_official.csv'))   

dataMainDirectory = '/datadrive/FEADD/'
fer67_cnn, incV3_cnn = importModels(dataMainDirectory)

for ff in range(opt.initial, opt.final + 1):

  basePath = dataMainDirectory + 'PNG/dfdc_train_part_' + str(ff) + '/'
  output_path = dataMainDirectory + 'CSV_OUT/dfdc_train_part_' + str(ff) + '/' 
  os.makedirs(os.path.join(output_path, 'REAL'), exist_ok=True)
  os.makedirs(os.path.join(output_path, 'FAKE'), exist_ok=True)
  videos = os.listdir(basePath)
  print(basePath)
  print(output_path)

  with open(dataMainDirectory + 'MetadataTrain/' +  str(ff) + '_zip_metadata.json', 'r') as f:
    test_df = json.load(f)
    
  for video in videos:
    #label = test_df.loc[test_df['filename'] == video + '.mp4']['label'].values[0]
  
    label = test_df[video + '.mp4']['label']
    video_sub = os.listdir(basePath + video)
  
    for subject in video_sub:
      print('video ' + video + '-' + subject + '-' + label)
      num= len(os.listdir(basePath + video + '/' + subject))
    
      frames_299 = []
      frames_48 = []
      idx_order = []
      single_images = os.listdir(basePath + video + '/' + subject)
    
      try:
    
        for file_png in single_images:
          idx_order.append(int(file_png.split('_')[0]))
      
          x = cv2.imread(basePath + video + '/' + subject + '/' + file_png)
          x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
          x_299 = cv2.resize(x, (299, 299),interpolation=cv2.INTER_LINEAR)
      
          y = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
          y_48= cv2.resize(y, (48, 48),interpolation=cv2.INTER_LINEAR)
      
          frames_299.append(x_299)
          frames_48.append(y_48)
     
        frames_299 = np.array(frames_299)
        frames_48 = np.array(frames_48)
    
        merged_df_fer67_IncV3 = features_extraction_step2(frames_48, frames_299, fer67_cnn, incV3_cnn )  
        merged_df_fer67_IncV3['label_y']= [label]*num
        merged_df_fer67_IncV3['order'] = idx_order
        merged_df_fer67_IncV3 = merged_df_fer67_IncV3.sort_values('order')
    
        if merged_df_fer67_IncV3.shape[0] > 30:
          if label == 'REAL':
            #merged_df_fer67_IncV3.to_pickle(os.path.join(output_path, 'REAL', video+'_'+subject))
            merged_df_fer67_IncV3.to_csv(os.path.join(output_path, 'REAL', video+'_'+subject), index = False)
          else:
            #merged_df_fer67_IncV3.to_pickle(os.path.join(output_path, 'FAKE', video+'_'+subject))  
            merged_df_fer67_IncV3.to_csv(os.path.join(output_path, 'FAKE', video+'_'+subject), index = False)
      
          with open(dataMainDirectory + 'PNG/dfdc_train_part_' + str(ff) + '_status.txt', "a") as file_object:
           file_object.write(video+'_'+subject +' ok\n')    
        else:
          print('Skipped because too small')
          with open(dataMainDirectory + 'PNG/dfdc_train_part_' + str(ff) + '_status.txt', "a") as file_object:
           file_object.write(video+'_'+subject +' skipped\n') 
      except:
        print(video+'_'+subject + ' errore')
        with open(dataMainDirectory + 'PNG/dfdc_train_part_' + str(ff) + '_status.txt', "a") as file_object:
          file_object.write(video+'_'+subject +' exception\n')