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

extracted_audio_features_df=pd.read_csv("./Extracted_Features.csv", sep = ",", names = ('feature','class'))
print("liSTA??\n\n")
print(extracted_audio_features_df)
print("\n\n")
X=np.array(extracted_audio_features_df['feature'].tolist())

y=np.array(extracted_audio_features_df['class'].tolist())
print(X)
print("\n\n\n\n\n\nGIORGIO\n\n\n\n\n\n")


labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
print(y)

audio_file = 'record.flac'
model=tf.keras.models.load_model("Detection_Model3.h5")
print("ciao")
audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
print(mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)
predicted_label=model.predict(mfccs_scaled_features)
print(predicted_label)
classes_x=np.argmax(predicted_label,axis=1)
prediction_class = labelencoder.inverse_transform(classes_x)
print(prediction_class)