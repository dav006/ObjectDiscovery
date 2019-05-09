""" 10_transfer_extract.py
Simple tranfer knowledge (feature extraction) using VGG16 for Cifar10.

UNAM IIMAS
Course:     Intro to Deep Learning
Professor:  Gibran Fuentes Pineda
Assistant:  Berenice Montalvo Lezama

Copyright (C) 2018 Berenice Montalvo Lezama
                   Ricardo Montalvo Lezama

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import datetime
import os
import time


import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense
from tensorflow.keras.models import Model, Sequential
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]


def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)

DATA_DIR = "../../annotations_landmarks_clean_struct"
HEIGHT = 250
WIDTH = 250
BATCH_SIZE = 16
NUM_CLASSES=585

datagen =  ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.2)

train_generator = datagen.flow_from_directory(DATA_DIR, 
                                                    target_size=(HEIGHT, WIDTH), 
						    interpolation='bicubic',
                                                    batch_size=BATCH_SIZE,
						    subset='training')

valid_generator = datagen.flow_from_directory(DATA_DIR,
                                                    target_size=(HEIGHT, WIDTH),
						    interpolation='bicubic',
                                                    batch_size=BATCH_SIZE,
						    subset='validation')

train_crops = crop_generator(train_generator, 224)
valid_crops = crop_generator(valid_generator, 224)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def timestamp():
  return datetime.datetime.fromtimestamp(time.time()).strftime('%y%m%d-%H%M%S')


def download():
  """ Download Cifar10 and Resnet50. """
  ResNet50(weights='imagenet', include_top=False)

def extract_train():
  """ Extract and train in the same model. """
  resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
  
  fc = GlobalAveragePooling2D()(resnet50.output)
  fc = Dense(NUM_CLASSES, activation='softmax', name='fc')(fc)
  model = Model(resnet50.input, fc)

  for layer in resnet50.layers:
    layer.trainable = False

  print('RestNet50-FC-1024-582')
  model.summary(line_length=100)
  model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

  
  print('Finish Resize')

  tensorboard = TensorBoard(
    log_dir='./graphs/transfer_learning' + timestamp(), histogram_freq=0, write_graph=True, write_images=False)

  model.fit_generator(train_crops,steps_per_epoch=BATCH_SIZE, epochs=200,validation_data = valid_crops,validation_steps = BATCH_SIZE, callbacks=[tensorboard])

  #evaluation = model.evaluate(X_tst, Y_tst)
  #print(evaluation)


def main(args):
  err_msg = 'Unknown function, options: extract, train, extract_train'
  if len(args) > 1:
    func_name = args[1]
    if func_name == 'download':
      download()
    elif func_name == 'extract':
      extract()
    elif func_name == 'train':
      train()
    elif func_name == 'extract_train':
      extract_train()
    else:
      print(err_msg)
  else:
    print(err_msg)
  return 0


if __name__ == '__main__':
  import sys
  sys.exit(main(sys.argv))
