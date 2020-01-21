from keras.layers import Dense
from keras.layers import Flatten
from Model import Model


class ANNModel(Model.Model):

    def define_model(self):
        self.model.add(Dense(784, activation='tanh', input_shape=(40, 1)))
        self.model.add(Dense(512, activation='sigmoid'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(4, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
