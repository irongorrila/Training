import pandas as pd
import numpy as np

from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.utils.np_utils import  to_categorical
from keras.callbacks import EarlyStopping

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

import time

###############################################################################


df= pd.read_csv('Data Sets/mnist_train.csv')

X_train = np.asarray(df.drop('5', axis=1))
y_train = to_categorical(np.asarray(df['5']))


X_train.shape
y_train.shape


def create_model(opt='adam', act='relu'):

    model = Sequential()
    model.add(Dense(50, activation= act, input_shape=(784,)))
    model.add(Dense(50, activation= act))
    model.add(Dense(10, activation= 'softmax'))

    model.compile(optimizer= opt, loss='categorical_crossentropy', \
        metrics=['accuracy'])

    return model


model= KerasClassifier(build_fn=create_model)

params = {  'epochs' :[3,5,7],
            'batch_size' : [5,10,20]}


rs = RandomizedSearchCV(model, params, cv=3)

a= time.time()
result= rs.fit(X_train, y_train)

b= time.time()
print('Best {} using {}'.format(rs.best_params_, rs.best_score_))

print(b-a)