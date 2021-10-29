from keras.applications import resnet50
from keras.preprocessing import image
import numpy as np


model = resnet50.ResNet50()

img = image.load_img('mykonos.jpg', target_size=(224, 224))

x= image.img_to_array(img)

x.shape
x= np.expand_dims(x, axis=0)

# scaling the input to match that in resnet

x= resnet50.preprocess_input(x)

pred = model.predict(x)

pred_classes = resnet50.decode_predictions(pred, top=10)

print('This is an image of:')

for imagenet_id, name, likelihood in pred_classes[0]:
    print('- {} :- {:2f}'.format(name, likelihood))