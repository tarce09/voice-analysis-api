from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from numpy import vectorize
import speech_recognition as sr
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


import json
import base64

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import librosa
import librosa.display
from matplotlib.pyplot import specgram
from sklearn.preprocessing import LabelEncoder
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

#--------------------------------------------
def extract_features(filename):
    audio, sample_rate = librosa.load(filename, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features



#--------------------------------------------
def getPrediction(filename):
    """
    This model return the predicted emotion of audio file
    """
    # Importin emotions
    emotions = pd.read_csv("emotions.csv")
    emotions = emotions["emotions"].tolist()
    # Importing model
    model = tf.keras.models.load_model("audio_classification3.hdf5")
    
    # Extracting features
    features = extract_features(filename)
    features = features.reshape(1, 40, 1)
    
    # Fit the labels to label encoder
    labelencoder = LabelEncoder()
    labelencoder.fit_transform(emotions)
    
    # Making predictions
    livepreds = model.predict(features, batch_size=32, verbose=1)
    livepreds1=livepreds.argmax(axis=1)
    liveabc = livepreds1.astype(int).flatten()
    livepredictions = (labelencoder.inverse_transform((liveabc)))
    print(livepredictions)
    
    return livepredictions




def index(request):
    response = json.dumps([{}])
    return HttpResponse(response, content_type='text/json')

#def get_car(request, car_name):
#    if request.method == 'GET':
#        try:
#            car = Car.objects.get(name=car_name)
#            response = json.dumps([{ 'Car': car.name, 'Top Speed': car.top_speed}])
#        except:
#            response = json.dumps([{ 'Error': 'No car with that name'}])
#    return HttpResponse(response, content_type='text/json')

#http://127.0.0.1:8000/Hello%20I%20am%20aryan%20i%20am%2020%20and%20i%20study%20in%20mit

# https://drive.google.com/uc?id=1r2-tsxk7zSA_RO_7gLciqsge4p1zLW4A


@api_view(['POST'])
def post_data(request):
    
    data=request.data
    
    buffer=(data["buffer"])
    
    audio_data=base64.b64decode(buffer)

    with open("file.wav",'wb') as pcm:
        pcm.write(audio_data)
        
    string=(data["string"])
 
    vsr=sr.Recognizer()
    with sr.AudioFile('file.wav') as source:
        audio = vsr.listen(source)

        try:
            text=vsr.recognize_google(audio)
            text=text.lower()
            string=string.lower()
            sentences=[text,string]
            vectorizer=CountVectorizer().fit_transform(sentences)
            vectors=vectorizer.toarray()
                
            csim=cosine_similarity(vectors)
            answer= csim[1][0]*100
            response = answer/10
            filepath="file.wav"
            emo_val=getPrediction(filepath)
            return Response({'status':response,'debug':text,'debug2':string,"value":emo_val})
        except:
            response = "420"
            
        
        return Response({'status':0,'debug':0,"value":0})   
    
            
            