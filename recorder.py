import os
import time

import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from CNN import CNN


def load_model(model_name, model_location):
    return CNN.get_model(model_name, model_location)


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
    print("Starting myrecording")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)

    sd.wait()  # Wait until recording is finished
    print("Finished Recording")
    filename = int(time.time()).__str__() + ".wav"
    write(os.path.join(location, filename), fs, myrecording)  # Save as WAV file
    return filename


def get_Emotion(num):
    emotion = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
    return emotion[num]


model_name = "cnn_SER.h5"
model_location = "C:/Users/ranja/PycharmProjects/VOICE-EMOTION-ANALYSIS/Model"

model = load_model(model_name, model_location)

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
