from keras.layers import Dense, LSTM
from keras.layers import Dropout, Activation

from Model import Model


class LSTMModel(Model.Model):

    def define_model(self):
        self.model.add(LSTM(100, input_shape=(40, 1)))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(4))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
