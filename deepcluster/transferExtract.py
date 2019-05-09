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
from tensorflow.keras.applications import vgg16
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from skimage.transform import resize


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def resize_image_arr(img_arr):
    x_resized_list = []
    for i in range(img_arr.shape[0]):
        img = img_arr[0]
        resized_img = resize(img, (48, 48))
        x_resized_list.append(resized_img)
    return np.stack(x_resized_list)

def timestamp():
  return datetime.datetime.fromtimestamp(time.time()).strftime('%y%m%d-%H%M%S')


def download():
  """ Download Cifar10 and VGG16. """
  tf.keras.datasets.cifar10.load_data()
  vgg16.VGG16(weights='imagenet', include_top=False)


def extract():
  """ Extract CNN codes to X_trn_cnn.npy, X_tst_cnn.npy. """
  vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
  flatten = Flatten()(vgg.output)
  model = Model(vgg.input, flatten)
  print('VGG16')
  model.summary(line_length=100)

  data = tf.keras.datasets.cifar10.load_data()
  (X_trn, _), (X_tst, _) = data

  X_trn_cnn = model.predict(X_trn)
  X_tst_cnn = model.predict(X_tst)

  np.save('X_trn_cnn.npy', X_trn_cnn)
  np.save('X_tst_cnn.npy', X_tst_cnn)


def train():
  """ Train VGG16 using CNN codes from X_trn_cnn.npy, X_tst_cnn.npy. """
  model = Sequential([
      Dense(128, input_shape=(512,), activation='relu', name='fc1'),
      Dropout(0.25, name='dp1'),
      Dense(10, activation='softmax', name='fc2')
  ])
  # model = Sequential([
  #     Dense(512, input_shape=(512,), activation='relu', name='fc1'),
  #     Dropout(0.25, name='dp1'),
  #     Dense(128, activation='relu', name='fc2'),
  #     Dropout(0.25, name='dp2'),
  #     Dense(10, activation='softmax', name='fc3')
  # ])
  print('FC-128-10')
  model.summary(line_length=100)
  model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
  )

  X_trn_cnn = np.load('X_trn_cnn.npy')
  X_tst_cnn = np.load('X_tst_cnn.npy')
  data = tf.keras.datasets.cifar10.load_data()
  (_, Y_trn), (_, Y_tst) = data
  model.fit(X_trn_cnn, Y_trn, epochs=100, batch_size=64)

  evaluation = model.evaluate(X_tst_cnn, Y_tst)
  print(evaluation)


def extract_train():
  """ Extract and train in the same model. """
  vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

  fc = Flatten()(vgg.output)
  fc = Dense(128, activation='relu', name='fc1')(fc)
  fc = Dropout(0.25, name='dp1')(fc)
  fc = Dense(10, activation='softmax', name='fc2')(fc)
  model = Model(vgg.input, fc)

  for layer in vgg.layers:
    layer.trainable = False

  print('VGG16-FC-128-10')
  model.summary(line_length=100)
  model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
  )

  data = tf.keras.datasets.cifar10.load_data()
  (X_trn, Y_trn), (X_tst, Y_tst) = data
  X_trn = resize_image_arr(X_trn)
  X_tst = resize_image_arr(X_tst)
  print('Finish Resize')

  tensorboard = TensorBoard(
    log_dir='./graphs/10_transfer_extract_' + timestamp(), 
    histogram_freq=0, write_graph=True, write_images=False)

  model.fit(X_trn, Y_trn, epochs=25, batch_size=64, callbacks=[tensorboard])

  evaluation = model.evaluate(X_tst, Y_tst)
  print(evaluation)


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
