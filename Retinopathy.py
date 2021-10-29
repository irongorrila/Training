import os
import pandas as pd
import matplotlib.pyplot as plt


from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

import cv2

#########################################################
# Importing the file paths in the csv file

df = pd.read_csv('Data Sets/photos_data/trainLabels.csv')
img_dir = '/Users/pc/Desktop/PGP Space/Data Sets/photos_data/dataset/'

df['path'] = img_dir + df.image + '.jpeg'

df.level.value_counts()

#########################################################
# checking to see if the paths have been added correctly
img= plt.imread(df.path[50])
img_n = img.reshape(256, 256, 3)
plt.imshow(img_n)
plt.show()

img_n.shape

# All files have a resolution of 2592 X 3888 pixels 

########################################################