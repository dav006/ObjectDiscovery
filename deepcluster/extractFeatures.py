import datetime
import os
import time


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

# defining a function to read images
def read_img(img_path):
    img = image.load_img(img_path, target_size=(HEIGHT, WIDTH, 3))
    img = image.img_to_array(img)

    return img

def extract_features(img_path,sample_count):
    features = np.zeros(shape=(sample_count, 2048))

def test_gen(X_test,datagen):
    Y_trash = np.ones(X_test.shape[0])
    flow = datagen.flow(X_test, Y_trash)
    for X,Y in flow:
        yield X #ignore Y

datagen =  ImageDataGenerator(preprocessing_function=preprocess_input)
data_generator = datagen.flow_from_directory(IMAGE_PATH,
                                                    target_size=(HEIGHT, WIDTH),
						    interpolation='bicubic',
                                                    batch_size=1000,
						    class_mode='categorical',
                                                    shuffle=False,
                                                    color_mode='rgb')

#Load images
'''
data_img = [None]*2000
index = 0
for img_path in sorted(os.listdir(IMAGE_PATH+'.')):
    data_img[index]=read_img(IMAGE_PATH + img_path)
    index+=1
    if index%2000 is 0:
       print index
       break

X_data = np.array(data_img, np.float32)
'''

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

model = load_model('finetunned.h5')
model2 = Model(model.input, model.get_layer('global_average_pooling2d').output)

X_data_gap = np.zeros(shape=(1000, 2048))

i=0
for inputs_batch,labels_batch in data_generator:
    X_data_gap = model2.predict(inputs_batch)
    break
'''
X_data_gap = model2.predict_generator(data_generator,steps=1)
'''
np.save('X_data_gap.npy', X_data_gap)
