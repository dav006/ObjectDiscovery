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
from sklearn.preprocessing import MultiLabelBinarizer
sys.path.append('../common/')
import smhHelper
import selectModels

IMAGE_PATH = '../../google_landmark_selected_crop/'
HEIGHT = 224
WIDTH = 224
NUM_CLASSES = 40
TOTAL_IMAGES = 20000
BATCH_SIZE = 20
EPOCHS = 10

CORPUS_FILE = 'google.corpus'
INVERT_INDEX_FILE = 'google.ifs'
MODEL_FILE = 'google.model'

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
one_hot = MultiLabelBinarizer(classes=NUM_CLASSES)
datagen =  ImageDataGenerator(preprocessing_function=preprocess_input)
data_generator = datagen.flow_from_directory(IMAGE_PATH,
                                                    target_size=(HEIGHT, WIDTH),
                                                    interpolation='bicubic',
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=False,
                                                    color_mode='rgb')

# Initial extraction of visual vocab and data creation
print 'Loading model'
start_time = time.time()
model1 = load_model('finetunned.h5')
model2 = Model(model1.input, model1.get_layer('global_average_pooling2d').output)
print('Loading model time: %.3f s' % (time.time() - start_time))
sys.stdout.flush()

X_data_gap = np.zeros(shape=(TOTAL_IMAGES, 2048))
print 'Predict data'
start_time = time.time()
for inputs_batch,labels_batch in data_generator:
    idx = (data_generator.batch_index - 1) * data_generator.batch_size
    if idx<0:
       idx = TOTAL_IMAGES-data_generator.batch_size
       X_data_gap[idx:idx+data_generator.batch_size] = model2.predict(inputs_batch)
       break
    #if idx+data_generator.batch_size>TOTAL_IMAGES:
    #    break
    # print idx
    sys.stdout.flush()
    X_data_gap[idx:idx+data_generator.batch_size] = model2.predict(inputs_batch)

data_generator.reset()
print('Predict data time: %.3f s' % (time.time() - start_time))

del model2
# Save to .corpus
print 'Create corpus'
sys.stdout.flush()
save_corpus(CORPUS_FILE,X_data_gap)
# Create inverted index
print 'Create inverted index'
sys.stdout.flush()
smhHelper.createInvertedIndex(CORPUS_FILE,INVERT_INDEX_FILE)
# Create model
print 'Create model'
start_time = time.time()
smhHelper.createModel(CORPUS_FILE,INVERT_INDEX_FILE,MODEL_FILE)
print('Create model time: %.3f s' % (time.time() - start_time))
sys.stdout.flush()

# Create ranking file
start_time = time.time()
Y_data = selectModels.selectModelsMulti(NUM_CLASSES,MODEL_FILE,INVERT_INDEX_FILE,TOTAL_IMAGES)
print('Select models time: %.3f s' % (time.time() - start_time))
sys.stdout.flush()

# Evaluate MAPP
start_time = time.time()
execfile('rankingImages.py')
print('ranking images time: %.3f s' % (time.time() - start_time))
sys.stdout.flush()

start_time = time.time()
execfile('evaluateGoogleThread.py')
print('Evaluate time: %.3f s' % (time.time() - start_time))
sys.stdout.flush()

# Create Model

fc=Dense(NUM_CLASSES, activation='sigmoid', name='fc')(model1.layers[-2].output)
#model3 = Model(inputs=model1.input,outputs=[fc,model1.get_layer('global_average_pooling2d').output])
model3 = Model(inputs=model1.input,outputs=fc)

layerFlag = False
for layer in model1.layers:
    if layer.name == 'res5a_branch2a':
        layerFlag = True
    layer.trainable = layerFlag

model3.summary(line_length=100)
model3.compile(
    optimizer=Adam(lr=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
  )

accuracyHistory = [None]*EPOCHS
#lossHistory = [None]*EPOCHS
# Train
for i in range(EPOCHS):
    epoch_time = time.time()
    accuracyEpoch = 0.0
    lossEpoch = 0.0
    Y_data = one_hot.fit_transform(Y_data)
    print 'Start train epoch : '+str(i)
    '''
    history = model3.fit_generator(data_generator,steps_per_epoch=data_generator.samples//BATCH_SIZE, epochs=1)
    '''
    for inputs_batch,labels_batch in data_generator:
    	idx = (data_generator.batch_index - 1) * data_generator.batch_size
        if idx<0:
           idx = TOTAL_IMAGES-data_generator.batch_size
           #print idx
           history = model3.fit(inputs_batch, np.array(Y_data[idx : idx + data_generator.batch_size]), batch_size=BATCH_SIZE, epochs=1,verbose=0)
           accuracyEpoch += history.history['acc']
          # loss += history.history['loss']
           sys.stdout.flush()
           break

        #if idx+data_generator.batch_size>TOTAL_IMAGES:
        #   break
        # print idx
        history = model3.fit(inputs_batch, np.array(Y_data[idx : idx + data_generator.batch_size]), batch_size=BATCH_SIZE, epochs=1,verbose=0)
        accuracyEpoch += history.history['acc'][0]
        #loss += history.history['loss'][0]
    
    data_generator.reset()
    accuracyHistory[i] = accuracyEpoch / (float(TOTAL_IMAGES)/float(BATCH_SIZE))
    #lossHistory[i] = lossEpoch / (float(TOTAL_IMAGES)/float(BATCH_SIZE))
    print 'Accuracy : '+str(accuracyHistory[i])
    #print 'Loss : '+str(lossHistory[i])

    # Extract data from GAP
    print 'Predict epoch'
    start_time = time.time()
    sys.stdout.flush()
    model4 = Model(model3.input, model3.get_layer('global_average_pooling2d').output)
    for inputs_batch,labels_batch in data_generator:
        idx = (data_generator.batch_index - 1) * data_generator.batch_size
        if idx<0:
           idx = TOTAL_IMAGES-data_generator.batch_size
           #print idx
           #sys.stdout.flush()
           X_data_gap[idx:idx+data_generator.batch_size] = model4.predict(inputs_batch)
           break
        #if idx+data_generator.batch_size>TOTAL_IMAGES:
        #   break
        # print idx
        X_data_gap[idx:idx+data_generator.batch_size] = model4.predict(inputs_batch)
    del model4
    data_generator.reset()
    print('Predict time: %.3f s' % (time.time() - start_time))

    # Save to .corpus
    print 'Create corpus'
    sys.stdout.flush()
    save_corpus(CORPUS_FILE,X_data_gap)
    
    # Create inverted index
    print 'Create inverted index'
    smhHelper.createInvertedIndex(CORPUS_FILE,INVERT_INDEX_FILE)
    
    # Create model
    print 'Create model'
    start_time = time.time()
    smhHelper.createModel(CORPUS_FILE,INVERT_INDEX_FILE,MODEL_FILE)
    print('Create model time: %.3f s' % (time.time() - start_time))
    sys.stdout.flush()

    # Create ranking file
    start_time = time.time()
    Y_data = selectModels.selectModelsMulti(NUM_CLASSES,MODEL_FILE,INVERT_INDEX_FILE,TOTAL_IMAGES)
    print('Select models time: %.3f s' % (time.time() - start_time))
    sys.stdout.flush()

    # Evaluate MAPP
    start_time = time.time()
    execfile('rankingImages.py')
    print('ranking images time: %.3f s' % (time.time() - start_time))
    sys.stdout.flush()
 
    start_time = time.time()
    execfile('evaluateGoogleThread.py')
    print('Evaluate time: %.3f s' % (time.time() - start_time))
    sys.stdout.flush()

    print('Epoch time: %.3f s' % (time.time() - epoch_time))
    sys.stdout.flush()
print accuracyHistory
#print lossHistory
