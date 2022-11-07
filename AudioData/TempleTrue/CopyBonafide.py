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

audio_data = pd.read_csv("ASVspoof2019.LA.cm.train.trn.txt", sep = " ", names = ('Id','Filename','a','FakeType','Class'))
print(audio_data)


print(audio_data['Class'].value_counts())



audio_data_frame = audio_data[audio_data['FakeType'] == '-']
print(audio_data_frame.shape)


Path('AudioData/Human_voice').mkdir(parents = True ,exist_ok = True)
real_voice = audio_data_frame['Filename']

for file in real_voice:
  fileName = os.path.join('./LA/ASVspoof2019_LA_train/flac/',file +'.flac')
  if os.path.isfile(fileName):
    copy(fileName,'AudioData/Human_voice')

for file in real_voice:
  fileName = os.path.join('./LA/ASVspoof2019_LA_dev/flac/',file +'.flac')
  if os.path.isfile(fileName):
    copy(fileName,'AudioData/Human_voice')


audio_data_frame = audio_data[audio_data['FakeType'] != '-']
print(audio_data_frame.shape)







