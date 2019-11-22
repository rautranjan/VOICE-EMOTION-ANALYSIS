import os
import time

import keras
import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write


def load_model(model_name, model_location):
    model = keras.models.load_model(os.path.join(model_location, model_name))
    model.summary()
    return model


def reformat_data(file):
    X = []
    arr, sampling_rate = librosa.load(file)
    mfcc_values = np.array(np.mean(librosa.feature.mfcc(y=arr, sr=sampling_rate, n_mfcc=40).T, axis=0))
    X.append(mfcc_values)
    X = np.array(X)
    X = np.expand_dims(X, axis=2)
    return X


def record_speech(location="recordings"):
    if not os.path.isdir(location):
        os.makedirs(location)
    fs = 22050  # Sample rate
    seconds = 3  # Duration of recording
    print("Starting Recording")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)

    sd.wait()  # Wait until recording is finished
    print("Finished Recording")
    filename = int(time.time()).__str__() + ".wav"
    write(os.path.join(location, filename), fs, myrecording)  # Save as WAV file
    return filename


def get_Emotion(num):
    emotion = ["Neutral", "Sad", "Happy", "Angry"]
    return emotion[num]


lstm_model_name = "LSTM.h5"
cnn_model_name = "CNN.h5"
model_location = "C:/Users/ranja/PycharmProjects/VOICE-EMOTION-ANALYSIS/ModelStore"


model_type  = int(input("\nPlease select type of Model from below options :\n\t1) CNN\t2) LSTM\n"))

if model_type==1:
    model = load_model(cnn_model_name, model_location)
    print("CNN Model has been Selected")
else:
    model = load_model(lstm_model_name, model_location)
    print("LSTM Model has been Selected")


print("\n--------------------------------------------------\n")
location = "recordings"
num = -1
while 1:
    print("\n\nPlease select from below options")
    num = int(input("Enter any number : Record your voice\nEnter 2 : Exit\n"))
    if num == 2:
        break

    filename = record_speech(location)
    X = reformat_data(os.path.join(location, filename))
    pred = model.predict_classes(X)

    print("Emotion : ", get_Emotion(int(pred)))
