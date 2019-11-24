from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten, Dropout, Activation


from Model import Model


class CNNModel(Model.Model):

    def define_model(self):
        self.model.add(Conv1D(128, 5, padding='same', input_shape=(40, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.1))
        self.model.add(MaxPooling1D(pool_size=(8)))
        self.model.add(Conv1D(128, 5, padding='same', ))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.1))
        self.model.add(MaxPooling1D(pool_size=(2)))
        self.model.add(Flatten())
        self.model.add(Dense(4))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
