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
'''
audio_data = pd.read_csv(r"ASVspoof2019.LA.cm.train.trn.txt", sep = " ", names = ('Id','Filename','a','FakeType','Class'))
audio_data_frame = audio_data[audio_data['FakeType'] == '-']


#INSERT THE FILES IN THE SPECIF FOLDER

Path('AudioData/Human_voice').mkdir(parents = True ,exist_ok = True)
real_voice = audio_data_frame['Filename']

for file in real_voice:
  fileName = os.path.join('C:/Users/marco/OneDrive/Desktop/progettoCyber/Deepfake-Audio-Detection-main/LA/ASVspoof2019_LA_train/flac/',file +'.flac')
  if os.path.isfile(fileName):
    copy(fileName,'AudioData/Human_voice')


audio_data_frame = audio_data[audio_data['FakeType'] != '-']
print(audio_data_frame.shape)


Path('AudioData/Generated_voice').mkdir(parents = True ,exist_ok = True)
fake_voice = audio_data_frame['Filename']
count=0
for file in fake_voice:
  fileName = os.path.join('C:/Users/marco/OneDrive/Desktop/progettoCyber/Deepfake-Audio-Detection-main/LA/ASVspoof2019_LA_train/flac/',file +'.flac')
  if os.path.isfile(fileName):
    copy(fileName,'AudioData/Generated_voice')
    count=count+1
    if count==12483:
      break'''


##EXTRACT THE FEATURES FROM FILE
directoryRealData=os.path.abspath(r".\..\..\progettoCyber\Deepfake-Audio-Detection-main\AudioData\Human_voice")
##for real data
def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

extracted_audio_features=[]
for filename in os.listdir(directoryRealData):
      final_class_labels="bonafide"
      data=features_extractor(directoryRealData+"\\"+filename)
      extracted_audio_features.append([data,final_class_labels])


 
#extracted_audio_features_df=pd.DataFrame(extracted_audio_features,columns=['feature','class'])

##for fake data
directoryFakeData=os.path.abspath(r".\..\..\progettoCyber\Deepfake-Audio-Detection-main\AudioData\Generated_voice")
for filename in os.listdir(directoryRealData):
      final_class_labels="spoof"
      data=features_extractor(filename)
      extracted_audio_features.append([data,final_class_labels])



extracted_audio_features_df=pd.DataFrame(extracted_audio_features,columns=['feature','class'])



#Saving Extracted_Features to file
with open("./Extracted_Features.txt","wb") as binary_file:
  pickle.dump(extracted_audio_features_df,binary_file)
binary_file.close()