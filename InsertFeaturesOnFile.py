import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import csv
import soundfile as sf
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from shutil import copy
import pickle

import tensorflow as tf
from random import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import os.path
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

audio_data = pd.read_csv(r"ASVspoof2019.LA.cm.eval.trl.txt", sep = " ", names = ('Id','Filename','a','FakeType','Class'))
audio_data_frame = audio_data[audio_data['FakeType'] == '-']

'''
#INSERT THE FILES IN THE SPECIF FOLDER

Path('AudioData/Human_voice').mkdir(parents = True ,exist_ok = True)
real_voice = audio_data_frame['Filename']

for file in real_voice:
  fileName = os.path.join('./LA/ASVspoof2019_LA_eval/flac/',file +'.flac')
  if os.path.isfile(fileName):
    copy(fileName,'AudioData/Human_voice')


audio_data_frame = audio_data[audio_data['FakeType'] != '-']
print(audio_data_frame.shape)


Path('AudioData/Generated_voice').mkdir(parents = True ,exist_ok = True)
fake_voice = audio_data_frame['Filename']

for file in fake_voice:
  fileName = os.path.join('./LA/ASVspoof2019_LA_eval/flac/',file +'.flac')
  if os.path.isfile(fileName):
    copy(fileName,'AudioData/Generated_voice')
    '''


##EXTRACT THE FEATURES FROM FILE




##for real data
def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

extracted_audio_features=[]
for index_num,row in tqdm(audio_data.iterrows()):
    file_name = os.path.join(os.path.abspath(r'C:\Users\marco\OneDrive\Desktop\progettoCyber\Deepfake-Audio-Detection-main\AudioData\Human_voice'),str(row["Filename"]) +'.flac')#MODIFY
    file_exists = os.path.exists(file_name)

    if file_exists:
      final_class_labels=row["Class"]
      data=features_extractor(file_name)
      extracted_audio_features.append([data,final_class_labels])
    else:
      continue


 
extracted_audio_features_df=pd.DataFrame(extracted_audio_features,columns=['feature','class'])

##for fake data

def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

for index_num,row in tqdm(audio_data.iterrows()):
    file_name = os.path.join(os.path.abspath(r'C:\Users\marco\OneDrive\Desktop\progettoCyber\Deepfake-Audio-Detection-main\AudioData\Generated_voice'),str(row["Filename"]) +'.flac')#MODIFY
    file_exists = os.path.exists(file_name)

    if file_exists:
      final_class_labels=row["Class"]
      data=features_extractor(file_name)
      extracted_audio_features.append([data,final_class_labels])
    else:
      continue



extracted_audio_features_df=pd.DataFrame(extracted_audio_features,columns=['feature','class'])
X=np.array(extracted_audio_features_df['feature'].tolist())
y=np.array(extracted_audio_features_df['class'].tolist())


labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=0)


#Saving Extracted_Features to file
with open("./Extracted_Features.txt","wb") as binary_file:
  pickle.dump(extracted_audio_features_df,binary_file)
binary_file.close()