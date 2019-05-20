import datetime
import os
import time
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras.layers import Dense, Input, Layer,Dropout
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
sys.path.append('../common/')
import smhHelper
import selectModels

IMAGE_PATH = '../../google_landmark_selected_crop/'
HEIGHT = 224
WIDTH = 224
NUM_CLASSES = 40
TOTAL_IMAGES = 1000

CORPUS_FILE = 'google.corpus'
INVERT_INDEX_FILE = 'google.ifs'
MODEL_FILE = 'google.model'

# defining a function to read images
def read_img(img_path):
    img = image.load_img(img_path, target_size=(HEIGHT, WIDTH, 3))
    img = image.img_to_array(img)

    return img

def save_corpus(corpus_file,X_data_gap):
    outputFile = open(corpus_file,'w')
    for X_data in X_data_gap:
        X_data_new = {}
	for wordId in range(0,len(X_data)):
		if int(X_data[wordId]) > 0:
			X_data_new[wordId] = X_data[wordId]
        outputFile.write(str(len(X_data_new)))
        for wordId in X_data_new:
                outputFile.write(' ')
                outputFile.write(str(wordId)+':1')
        outputFile.write('\n')
    outputFile.close
'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
'''
# Load data
print 'Creating data generators'
datagen =  ImageDataGenerator(preprocessing_function=preprocess_input)
data_generator = datagen.flow_from_directory(IMAGE_PATH,
                                                    target_size=(HEIGHT, WIDTH),
                                                    interpolation='bicubic',
                                                    batch_size=TOTAL_IMAGES,
                                                    class_mode='categorical',
                                                    shuffle=False,
                                                    color_mode='rgb')

# Initial extraction of visual vocab and data creation
print 'Loading model'
model1 = load_model('finetunned.h5')
model2 = Model(model1.input, model1.get_layer('global_average_pooling2d').output)

X_data_gap = np.zeros(shape=(TOTAL_IMAGES, 2048))
print 'Predict data'
i=0

input_batch_global = 0
for inputs_batch,labels_batch in data_generator:
    input_batch_global = inputs_batch
    X_data_gap = model2.predict(inputs_batch)
    break
del model2
# Save to .corpus
print 'Create corpus'
save_corpus(CORPUS_FILE,X_data_gap)
# Create inverted index
print 'Create inverted index'
smhHelper.createInvertedIndex(CORPUS_FILE,INVERT_INDEX_FILE)
# Create model
print 'Create model'
smhHelper.createModel(CORPUS_FILE,INVERT_INDEX_FILE,MODEL_FILE)

# Create ranking file
Y_data = selectModels.selectModels(NUM_CLASSES,MODEL_FILE,INVERT_INDEX_FILE,TOTAL_IMAGES)

# Evaluate MAPP

execfile('rankingImages.py')
execfile('evaluateGoogle.py')


# Create Model

fc=Dense(NUM_CLASSES, activation='softmax', name='fc')(model1.layers[-2].output)
#model3 = Model(inputs=model1.input,outputs=[fc,model1.get_layer('global_average_pooling2d').output])
model3 = Model(inputs=model1.input,outputs=fc)

layerIndex = 0
for layer in model1.layers:
    if layerIndex > 165:
	layer.trainable = True
    else:
        layer.trainable = False
    layerIndex+=1

model3.summary(line_length=100)
model3.compile(
    optimizer=Adam(lr=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )


# Train
for i in range(5):

    Y_data = to_categorical(Y_data,num_classes=NUM_CLASSES)
    history = model3.fit(input_batch_global, np.array(Y_data), batch_size=TOTAL_IMAGES, epochs=1)
    model4 = Model(model3.input, model3.get_layer('global_average_pooling2d').output)
    X_data_gap = model4.predict(inputs_batch)
    del model4

    # Save to .corpus
    print 'Create corpus'
    save_corpus(CORPUS_FILE,X_data_gap)
    # Create inverted index
    print 'Create inverted index'
    smhHelper.createInvertedIndex(CORPUS_FILE,INVERT_INDEX_FILE)
    # Create model
    print 'Create model'
    smhHelper.createModel(CORPUS_FILE,INVERT_INDEX_FILE,MODEL_FILE)
    # Create ranking file
    Y_data = selectModels.selectModels(NUM_CLASSES,MODEL_FILE,INVERT_INDEX_FILE,TOTAL_IMAGES)

    # Evaluate MAPP
    execfile('rankingImages.py')
    execfile('evaluateGoogle.py')
