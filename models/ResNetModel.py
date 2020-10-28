from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

IMG_WIDTH = 96
IMG_HEIGHT = 96
NUM_CLASSES = 3

def make_basic_block_layer(filter_num, blocks, stride=1):
  """
  Generate a block layer
  """
  res_block = tf.keras.Sequential()
  res_block.add(basic_block(filter_num, stride=stride))

  for _ in range(1, blocks):
    res_block.add(basic_block(filter_num, stride=1))

  return res_block

def basic_block(filter_num, stride=1):
  """
  Add a downsample layer depending on the  stride
  """
  model  = tf.keras.Sequential([
    layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, padding='same'),
    layers.BatchNormalization()
  ])

  if stride != 1:
    return tf.keras.Sequential([
      model,
      layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=stride),
      layers.BatchNormalization(),
      layers.ReLU()
    ])
  else:
    return model

def make_bottleneck_layer(filter_num, blocks, stride=1):
  """
  Generate a bottleneck layer
  """
  res_block = tf.keras.Sequential()
  res_block.add(bottleneck_layer(filter_num, stride=stride))

  for _ in range(1, blocks):
    res_block.add(bottleneck_layer(filter_num, stride=1))

  return res_block

def bottleneck_layer(filter_num, stride=1):
  """
  Add a downsample layer
  """
  model  = tf.keras.Sequential([
    layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=1, padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=1, padding='same'),
    layers.BatchNormalization()
  ])

  return tf.keras.Sequential([
    model,
    layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=stride),
    layers.BatchNormalization(),
    layers.ReLU()
  ])

def resnet_18(final_activation):
  nodes = [2, 2, 2, 2]
  model = tf.keras.Sequential([
      layers.Conv2D(16, 3, padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH , 3), strides=2),
      layers.BatchNormalization(),
      layers.ReLU(),
      layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'),
      make_basic_block_layer(filter_num=64, blocks=nodes[0]),
      make_basic_block_layer(filter_num=128, blocks=nodes[1], stride=2),
      make_basic_block_layer(filter_num=256, blocks=nodes[2], stride=2),
      make_basic_block_layer(filter_num=512, blocks=nodes[3], stride=2),
      layers.GlobalAveragePooling2D(),
      # layers.Dropout(0.1),
      layers.Dense(units=NUM_CLASSES, activation=final_activation, name='predictions')
  ])

  return model

def resnet_34(final_activation):
  nodes = [3, 4, 6, 3]
  model = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH , 3), strides=2),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'),
    make_bottleneck_layer(filter_num=64, blocks=nodes[0]),
    make_bottleneck_layer(filter_num=128, blocks=nodes[1], stride=2),
    make_bottleneck_layer(filter_num=256, blocks=nodes[2], stride=2),
    make_bottleneck_layer(filter_num=512, blocks=nodes[3], stride=2),
    layers.GlobalAveragePooling2D(),
    # layers.Dropout(0.1),
    layers.Dense(units=NUM_CLASSES, activation=final_activation, name='predictions')
  ])

  return model


def get_model_resnet(model='resnet_18', optimizer='adam', loss='binary_crossentropy', final_activation='softmax', metrics='accuracy'):
  """
  Return a basic model
  """
  if model == 'resnet18':
    model = resnet_18(final_activation=final_activation)
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
    return model
  elif model == 'resnet34':
    model = resnet_34(final_activation=final_activation)
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
    return model
