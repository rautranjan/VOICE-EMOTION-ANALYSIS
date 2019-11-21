import os

import keras
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from CNN_Model import CNNModel


class CNN:
    __X_train = []
    __X_test = []
    __y_train = []
    __y_test = []
    __model = CNNModel()

    def __init__(self):
        self

    def prepare_dataset(self, source_path):
        X = []
        y = []
        for voicePattern in os.listdir(source_path):
            for actor in os.listdir(os.path.join(source_path, voicePattern)):
                for file in os.listdir(os.path.join(source_path, voicePattern, actor)):
                    if int(file[7:8]) in [6,8]:
                        continue
                    elif int(file[7:8]) in [1,2]:
                        y.append(0)
                    elif int(file[7:8]) in [4]:
                        y.append(1)
                    elif int(file[7:8]) in [3]:
                        y.append(2)
                    elif int(file[7:8]) in [5,7]:
                        y.append(3)
                    arr, sampling_rate = librosa.load(os.path.join(source_path, voicePattern, actor, file))
                    mfcc_values = np.array(np.mean(librosa.feature.mfcc(y=arr, sr=sampling_rate, n_mfcc=40).T, axis=0))
                    X.append(mfcc_values)

        X = np.array(X)
        y = np.array(y)

        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(X, y, test_size=0.33,
                                                                                        random_state=42)
        self.__X_train = np.expand_dims(self.__X_train, axis=2)
        self.__X_test = np.expand_dims(self.__X_test, axis=2)
        print("-------------------------------Finished Preparing Dataset-------------------------------")
        return self.__X_train, self.__X_test, self.__y_train, self.__y_test

    def train_save_model(self, X_train, X_test, y_train, y_test, name, location, batch_size=16, epochs=1000):
        training_details = self.__model.train_model(X_train, y_train, X_test, y_test, batch_size, epochs)
        print("-------------------------------Model Training Finished-------------------------------")
        self.__model.save_model(name, location)
        return training_details

    @staticmethod
    def get_model(name, location):
        new_model = keras.models.load_model(os.path.join(location, name))
        print("Loaded Model at path ", os.path.join(location, name))
        return new_model

    @staticmethod
    def get_model_location(name, location):
        return os.path.join(location, name)


model_name = "cnn.h5"
model_location = "C:/Users/ranja/PycharmProjects/VOICE-EMOTION-ANALYSIS/Model"

if __name__ == "__main__":
    cnn = CNN()
    source_path = "C:/Users/ranja/PycharmProjects/VOICE-EMOTION-ANALYSIS/RAVDESS"
    X_train, X_test, y_train, y_test = cnn.prepare_dataset(source_path)

    training_details = cnn.train_save_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                            name=model_name, location=model_location, batch_size=16, epochs=1000)

    plt.plot(training_details.history['loss'])
    plt.plot(training_details.history['val_loss'])
    plt.title('Epoch vs Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.savefig('images/Epoch_vs_Loss.png')
    plt.close()

    plt.plot(training_details.history['accuracy'])
    plt.plot(training_details.history['val_accuracy'])
    plt.title('Epoch vs Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.savefig('images/Epoch_vs_Accuracy.png')
    plt.close()

    model = CNN.get_model(model_name, model_location)
    pred = model.predict_classes(X_test)
    c_report = classification_report(y_test, pred)
    print(c_report)

    c_matrix = confusion_matrix(y_test, pred)
    print(c_matrix)
