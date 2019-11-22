import abc
import os

from keras.models import Sequential
from keras.utils import to_categorical


class Model(abc.ABC):

    def __init__(self):
        self.model = Sequential()
        self.define_model()

    @abc.abstractmethod
    def define_model(self):
        pass

    def train_model(self, X_train, y_train, X_test, y_test, batch_size, epochs):
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        training_details = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                          validation_data=(X_test, y_test))

        return training_details

    def save_model(self, name, location):
        if not os.path.isdir(location):
            os.makedirs(location)
        path = os.path.join(location, name)
        self.model.save(path)
        print('Saved trained model at %s ' % path)
