import datetime
import os
import time
import sys

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MultiLabelBinarizer

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
import kmeans_ann_hns
import toBagOfWordsHSM_google

IMAGE_PATH = '../../google_landmark_selected_crop/'
HEIGHT = 224
WIDTH = 224
NUM_CLASSES = 100
TOTAL_IMAGES = 20000
BATCH_SIZE = 20
EPOCHS = 10
CONV_OUTPUT = 7
CONV_FEATURES = 980000
DIM=2048
PCA_DIM = 512
CONV_OUTPUT_POW=CONV_OUTPUT*CONV_OUTPUT

CORPUS_FILE = 'google.corpus'
INVERT_INDEX_FILE = 'google.ifs'
MODEL_FILE = 'google.model'

def convert_features_image(allDesc):

    finalAllDesc = np.zeros(shape=(TOTAL_IMAGES,CONV_OUTPUT_POW,PCA_DIM))
    index=0
    for desc in allDesc:
	indexImage = index/CONV_OUTPUT_POW
	indexDesc = index%CONV_OUTPUT_POW
        finalAllDesc[indexImage][indexDesc]= desc
        index+=1
    return finalAllDesc

def convert_features_row(allDesc):

    finalAllDesc = np.zeros(shape=(CONV_FEATURES,DIM))
    index=0
    for image in allDesc:
        for row in image:
                for desc in row:
                        finalAllDesc[index]= desc
                        index+=1
    return finalAllDesc


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
sys.stdout.flush()
start_time = time.time()
model1 = load_model('finetunned.h5')
model2 = Model(model1.input, model1.get_layer('activation_48').output)
print('Loading model time: %.3f s' % (time.time() - start_time))
sys.stdout.flush()

X_data_conv = np.zeros(shape=(TOTAL_IMAGES,CONV_OUTPUT,CONV_OUTPUT, 2048))
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
    # print idx
    sys.stdout.flush()
    X_data_conv[idx:idx+data_generator.batch_size] = model2.predict(inputs_batch)

data_generator.reset()
#X_data_conv = np.load('X_data_conv.npy')
X_data_conv = convert_features_row(X_data_conv)
print('Predict data time: %.3f s' % (time.time() - start_time))
del model2

#PCA
print 'Generate PCA'
sys.stdout.flush()
start_time = time.time()
X_data_conv = Normalizer(copy=False).fit_transform(X_data_conv)
print('Scaler time: %.3f s' % (time.time() - start_time))
sys.stdout.flush()
X_data_conv = PCA(n_components=PCA_DIM,whiten=True,copy=False).fit_transform(X_data_conv)
print('PCA time: %.3f s' % (time.time() - start_time))
sys.stdout.flush()
X_data_conv = Normalizer(copy=False).fit_transform(X_data_conv)
print('Scaler time: %.3f s' % (time.time() - start_time))
sys.stdout.flush()

# Generate visual vocabulary
print 'Generate Visual Vocabulary'
start_time = time.time()
kmeans_ann_hns.generateVocab(X_data_conv,PCA_DIM)
print('Generate visual vocab time: %.3f s' % (time.time() - start_time))
sys.stdout.flush()

# Save to .corpus
print 'Create corpus'
start_time = time.time()
X_data_conv = convert_features_image(X_data_conv)
toBagOfWordsHSM_google.createCorpus(X_data_conv,CORPUS_FILE)
print('Create corpus time: %.3f s' % (time.time() - start_time))
sys.stdout.flush()
del X_data_conv

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
Y_data = selectModels.selectModels(NUM_CLASSES,MODEL_FILE,INVERT_INDEX_FILE,TOTAL_IMAGES)
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
fc=Dense(NUM_CLASSES, activation='softmax', name='fc')(model1.layers[-2].output)
model3 = Model(inputs=model1.input,outputs=fc)

layerFlag = False
for layer in model1.layers:
    if layer.name == 'res5a_branch2a':
        layerFlag = True
    layer.trainable = layerFlag

#model3.summary(line_length=100)
model3.compile(
    optimizer=Adam(lr=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

accuracyHistory = [None]*EPOCHS
# Train
for i in range(EPOCHS):
    epoch_time = time.time()
    accuracyEpoch = 0.0
    lossEpoch = 0.0
    Y_data_keep = (Y_data > -1)
    Y_data = Y_data[Y_data_keep]
    Y_data = to_categorical(Y_data,num_classes=NUM_CLASSES)
    print 'Start train epoch : '+str(i)
    start_time = time.time()
    sys.stdout.flush()
    
    idy=0
    for inputs_batch,labels_batch in data_generator:
    	idx = (data_generator.batch_index - 1) * data_generator.batch_size
        if idx<0:
           idx = TOTAL_IMAGES-data_generator.batch_size
           Y_data_selected = Y_data_keep[idx : idx + data_generator.batch_size]
           #print idx
           inputs_batch = inputs_batch[Y_data_selected]
           history = model3.fit(inputs_batch, np.array(Y_data[idy : idy + inputs_batch.shape[0]]), batch_size=inputs_batch.shape[0], epochs=1,verbose=0)
           accuracyEpoch += history.history['acc']
           sys.stdout.flush()
           break
        
        Y_data_selected = Y_data_keep[idx : idx + data_generator.batch_size]
        inputs_batch = inputs_batch[Y_data_selected]
        history = model3.fit(inputs_batch, np.array(Y_data[idy : idy + inputs_batch.shape[0]]), batch_size=inputs_batch.shape[0], epochs=1,verbose=0)
        accuracyEpoch += history.history['acc'][0]
        idy+=inputs_batch.shape[0]
    
    print idy
    data_generator.reset()
    accuracyHistory[i] = accuracyEpoch / (float(TOTAL_IMAGES)/float(BATCH_SIZE))
    print 'Accuracy : '+str(accuracyHistory[i])
    print('Conv train time: %.3f s' % (time.time() - start_time))

    # Extract data from GAP
    print 'Predict epoch'
    start_time = time.time()
    sys.stdout.flush()
    X_data_conv = np.zeros(shape=(TOTAL_IMAGES,CONV_OUTPUT,CONV_OUTPUT, 2048))
    model4 = Model(model3.input, model3.get_layer('activation_48').output)
    for inputs_batch,labels_batch in data_generator:
        idx = (data_generator.batch_index - 1) * data_generator.batch_size
        if idx<0:
           idx = TOTAL_IMAGES-data_generator.batch_size
           X_data_conv[idx:idx+data_generator.batch_size] = model4.predict(inputs_batch)
           break
        X_data_conv[idx:idx+data_generator.batch_size] = model4.predict(inputs_batch)
    del model4
    data_generator.reset()
    X_data_conv = convert_features_row(X_data_conv)
    print('Predict time: %.3f s' % (time.time() - start_time))

    #PCA
    print 'Generate PCA'
    sys.stdout.flush()
    start_time = time.time()
    X_data_conv = Normalizer(copy=False).fit_transform(X_data_conv)
    print('Scaler time: %.3f s' % (time.time() - start_time))
    sys.stdout.flush()
    X_data_conv = PCA(n_components=PCA_DIM,whiten=True,copy=False).fit_transform(X_data_conv)
    print('PCA time: %.3f s' % (time.time() - start_time))
    sys.stdout.flush()
    X_data_conv = Normalizer(copy=False).fit_transform(X_data_conv)
    print('Scaler time: %.3f s' % (time.time() - start_time))
    sys.stdout.flush()

    # Generate visual vocabulary
    print 'Generate Visual Vocabulary'
    start_time = time.time()
    kmeans_ann_hns.generateVocab(X_data_conv,PCA_DIM)
    print('Generate visual vocab time: %.3f s' % (time.time() - start_time))
    sys.stdout.flush()

    # Save to .corpus
    print 'Create corpus'
    start_time = time.time()
    X_data_conv = convert_features_image(X_data_conv)
    toBagOfWordsHSM_google.createCorpus(X_data_conv,CORPUS_FILE)
    print('Create corpus time: %.3f s' % (time.time() - start_time))
    sys.stdout.flush()

    del X_data_conv
    
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
    Y_data = selectModels.selectModels(NUM_CLASSES,MODEL_FILE,INVERT_INDEX_FILE,TOTAL_IMAGES)
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
