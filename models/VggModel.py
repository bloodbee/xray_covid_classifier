from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

IMG_WIDTH = 96
IMG_HEIGHT = 96
NUM_CLASSES = 3

def get_model_vgg(model='vgg16', nodes=16, optimizer='adam', loss='binary_crossentropy', hidden_activation='linear', final_activation='softmax', metrics='accuracy'):
  """
  Return a VGG16 model - CF https://arxiv.org/pdf/1409.1556/
  """
  model = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation=hidden_activation, input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Conv2D(32, 3, padding='same', activation=hidden_activation),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Conv2D(64, 3, padding='same', activation=hidden_activation),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(512, activation=final_activation),
    layers.Dense(units=NUM_CLASSES)
  ])

  model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
  return model

