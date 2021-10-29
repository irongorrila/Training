from keras.layers import Input, Dense
from keras.utils.vis_utils import plot_model
from keras.models import Model

import matplotlib.pyplot as plt

input_tensor = Input(shape=(1,))

output_layer = Dense(1)
output_tensor = output_layer(input_tensor)

model = Model(input_tensor, output_tensor)

plot_model(model, to_file='model.png')

img= plt.imread('model.png')
plt.imshow(img)
plt.show()

