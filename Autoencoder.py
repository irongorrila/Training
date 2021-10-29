import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt


df= pd.read_csv('Data Sets/mnist_train.csv')

X_train = np.asarray(df.drop('5', axis=1))
y_train = to_categorical(np.asarray(df['5']))

X_test = X_train[500:1000,]

autoencoder = Sequential()

autoencoder.add(Dense(32, activation='relu', input_shape=(784,)))
autoencoder.add(Dense(784, activation='relu'))

autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')


encoder = Sequential()
encoder.add(autoencoder.layers[0])


plt.imshow(encoder.predict(X_test)[0].reshape(4,4))
plt.show()