import datetime
import os
import time
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_PATH = '../../google_landmark_selected_crop'
HEIGHT = 224
WIDTH = 224
BATCH_SIZE = 20
TOTAL_IMAGES = 20000
CONV_SIZE = 7

print 'Creating data generators'
datagen =  ImageDataGenerator(preprocessing_function=preprocess_input)
data_generator = datagen.flow_from_directory(IMAGE_PATH,
                                                    target_size=(HEIGHT, WIDTH),
                                                    interpolation='bicubic',
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=False,
                                                    color_mode='rgb')

model = load_model('finetunned.h5')
model2 = Model(model.input, model.get_layer('activation_48').output)

X_data_conv = np.zeros(shape=(TOTAL_IMAGES,CONV_SIZE,CONV_SIZE, 2048))

print 'Predict data'
start_time = time.time()
for inputs_batch,labels_batch in data_generator:
    idx = (data_generator.batch_index - 1) * data_generator.batch_size
    if idx<0:
       idx = TOTAL_IMAGES-data_generator.batch_size
       X_data_conv[idx:idx+data_generator.batch_size] = model2.predict(inputs_batch)
       break
    #if idx+data_generator.batch_size>TOTAL_IMAGES:
    #    break
    print idx
    sys.stdout.flush()
    X_data_conv[idx:idx+data_generator.batch_size] = model2.predict(inputs_batch)

data_generator.reset()
print('Predict data time: %.3f s' % (time.time() - start_time))

np.save('X_data_conv.npy', X_data_conv)
