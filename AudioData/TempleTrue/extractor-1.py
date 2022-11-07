

from tqdm import tqdm
import os.path
import librosa
import pandas as pd
import numpy as np

def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

file_name = "record.flac"
file_exists = os.path.exists(file_name)
extracted_audio_features=[]

final_class_labels="bonafide"
data=features_extractor(file_name)
extracted_audio_features.append([data,final_class_labels])
extracted_audio_features_df=pd.DataFrame(extracted_audio_features,columns=['feature','class'])

extracted_audio_features_df.to_csv(r'./recordFlac.csv', index=False)

file_name = "record2.flac"
file_exists = os.path.exists(file_name)
extracted_audio_features=[]

final_class_labels="bonafide"
data=features_extractor(file_name)
extracted_audio_features.append([data,final_class_labels])
extracted_audio_features_df=pd.DataFrame(extracted_audio_features,columns=['feature','class'])

extracted_audio_features_df.to_csv(r'./recordWav.csv', index=False)