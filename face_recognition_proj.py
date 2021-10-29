import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


#load dataset
data = np.load('Data Sets/ORL_faces.npz')

X_train = data['trainX']
X_test= data['testX']
Y_train= data['trainY']
Y_test= data['testY']



#normalizing
X_train = np.array(X_train,dtype='float32')/255
X_test= np.array(X_test,dtype='float32')/255

#checking
X_test.shape
Y_test.shape

train_x, test_x, train_y, test_y= train_test_split(X_train, Y_train, test_size=0.3, random_state=5)


im_rows=112
im_cols=92
im_shape=(im_rows, im_cols, 1)

#change the size of images
train_x = train_x.reshape(train_x.shape[0], *im_shape)
test_x = test_x.reshape(test_x.shape[0], *im_shape)


# Building the CNN

es = EarlyStopping(patience=5)

model= Sequential([
    Conv2D(50, kernel_size=7, activation='relu', input_shape= im_shape),
    MaxPooling2D(pool_size=2),
    Conv2D(75, kernel_size=5, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(1024, activation='relu'),
     Dropout(0.5),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    #20 is the number of outputs
    Dense(20, activation='softmax')  
])

model.compile(
    loss='sparse_categorical_crossentropy',#'categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

hist=model.fit(
    np.array(train_x), np.array(train_y), batch_size=200,
    epochs=100, verbose=2,validation_data=(np.array(test_x),np.array(test_y)))


# plot the result
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# plot loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()