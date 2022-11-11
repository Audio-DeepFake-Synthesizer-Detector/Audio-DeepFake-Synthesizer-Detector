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


#Retrieving Extracted_features from file
#VERSION LECTURE FROM BYTE FILE
with open("./Extracted_Features_Final.txt","rb") as binary_file:
  extracted_audio_features_df=pickle.load(binary_file)
binary_file.close()

print(extracted_audio_features_df)
X=np.array(extracted_audio_features_df['feature'].tolist())
y=np.array(extracted_audio_features_df['class'].tolist())


labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=0)
#Model Training information

rmsprop = RMSprop(learning_rate=0.001)

model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
model.summary()

#earlystop = EarlyStopping(patience=15)
lr_reduction = ReduceLROnPlateau(monitor = 'val_loss',
                                patience = 3,
                                verbose = 1,
                                factor = 0.2,
                                min_lr = 0.001
)

callbacks = [lr_reduction]

num_epochs = 50
num_batch_size = 32

# save the model
keras_file = "Detection_ModelFinal.h5"
tf.keras.models.save_model(model,keras_file)



history = model.fit(X_train,
          y_train,
          batch_size=num_batch_size,
          epochs=num_epochs,
          validation_data=(X_val, y_val),
          callbacks=callbacks,
          verbose=1)

#Plot a graph for Training accuracy and validation accuracy

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train','Validation'], loc='lower right')
plt.show()

#Plot a graph for Training accuracy and validation accuracy

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train','Validation'], loc='lower right')
plt.show()

#Plot a graph for Training Accuracy
plt.plot(history.history['accuracy'])
plt.title('Training Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Training Accuracy'], loc='lower right')
plt.show()

#Plot a graph for Validation Accuracy

plt.plot(history.history['val_accuracy'])
plt.title('Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Validation Accuracy'], loc='lower right')
plt.show()

#Plot a graph for Training Loss

plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training Loss'], loc='upper left')
plt.show()

#Plot a graph for Validation Loss

plt.plot(history.history['val_loss'])
plt.title('Validation Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Validation Loss'], loc='upper right')
plt.show()

##Model Prediction
test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print('Accuracy = ',test_accuracy[1])
print('Loss = ',test_accuracy[0])

##SAVE THE MODEL
tf.keras.models.save_model(model,keras_file)

##Testing the model
audio_file = 'testFile.wav'

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