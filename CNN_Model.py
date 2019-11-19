import os

import keras
import keras_metrics
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten, Dropout, Activation
from keras.models import Sequential


class CNNModel:
    __model = Sequential()

    def __init__(self):
        self.__define_model()

    def __define_model(self):
        self.__model.add(Conv1D(128, 5, padding='same', input_shape=(40, 1)))
        self.__model.add(Activation('relu'))
        self.__model.add(Dropout(0.1))
        self.__model.add(MaxPooling1D(pool_size=(8)))
        self.__model.add(Conv1D(128, 5, padding='same', ))
        self.__model.add(Activation('relu'))
        self.__model.add(Dropout(0.1))
        self.__model.add(Flatten())
        self.__model.add(Dense(8))
        self.__model.add(Activation('softmax'))
        # opt = keras.optimizers.rmsprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)
        # self.__model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
        #                    metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(),
        #                             keras_metrics.f1_score()])
        self.__model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
                             # metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(),
                             #          keras_metrics.f1_score()])
        self.__model.summary()

    def train_model(self, X_train, y_train, X_test, y_test, batch_size, epochs):
        training_details = self.__model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                          validation_data=(X_test, y_test))

        return training_details

    def save_model(self, name, location):
        if not os.path.isdir(location):
            os.makedirs(location)
        path = os.path.join(location, name)
        self.__model.save(path)
        print('Saved trained model at %s ' % path)
