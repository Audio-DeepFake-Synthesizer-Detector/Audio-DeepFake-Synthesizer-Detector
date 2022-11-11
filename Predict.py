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
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os.path
import pickle

#Extract txt
with open("./Extracted_Features_Final.txt","rb") as binary_file:
  extracted_audio_features_df=pickle.load(binary_file)
binary_file.close()

X=np.array(extracted_audio_features_df['feature'].tolist())
y=np.array(extracted_audio_features_df['class'].tolist())

labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
model=tf.keras.models.load_model("Detection_ModelFinal.h5")

#Directory for fake data
directoryTestFake = r'.\AudioData\SpoofTest'
fileNumber=0
goodPredict=0
spoof=['spoof']
for filename in os.listdir(directoryTestFake):
    f = os.path.join(directoryTestFake, filename)
    audio, sample_rate = librosa.load(f, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=model.predict(mfccs_scaled_features)
    print(predicted_label)
    classes_x=np.argmax(predicted_label,axis=1)
    prediction_class = labelencoder.inverse_transform(classes_x)
    print(prediction_class)
    fileNumber=fileNumber+1
    if prediction_class==spoof:
        goodPredict=goodPredict+1

print("Accuracy Fake files: "+str(goodPredict)+"/"+str(fileNumber)+"\n\n\n\n")

print("-------------------------------------------")

#Directory for true data
directoryTestFake = r'.\AudioData\BonafideTest'
fileNumber=0
goodPredict=0
bonafide=['bonafide']
for filename in os.listdir(directoryTestFake):
    f = os.path.join(directoryTestFake, filename)
    audio, sample_rate = librosa.load(f, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=model.predict(mfccs_scaled_features)
    print(predicted_label)
    classes_x=np.argmax(predicted_label,axis=1)
    prediction_class = labelencoder.inverse_transform(classes_x)
    print(prediction_class)
    fileNumber=fileNumber+1
    if prediction_class==bonafide:
        goodPredict=goodPredict+1

print("Accuracy Real files: "+str(goodPredict)+"/"+str(fileNumber))

