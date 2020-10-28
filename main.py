# Useful librairies
from __future__ import absolute_import, division, print_function, unicode_literals
import sys, time, datetime, shutil, os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import csv
import cv2
import matplotlib.pyplot as plt

# Our model
from models.VggModel import get_model_vgg
from models.ResNetModel import get_model_resnet


# Global variables
BATCH_SIZE = 32
NB_EPOCHS = 100
IMG_WIDTH = 96
IMG_HEIGHT = 96
TRAIN_DATA_PATH = 'chest_xray/train'
TEST_DATA_PATH = 'chest_xray/test'
VAL_DATA_PATH = 'chest_xray/val'
CLASS_NAMES = ['NORMAL', 'BACTERIA', 'VIRUS']
AUTOTUNE = tf.data.experimental.AUTOTUNE
VERBOSE = 1
# vgg16 or resnet18 or resnet34
MODEL_NAME = 'resnet34'
OPTIMIZER = 'rmsprop'
HIDDEN_ACTIVATION = 'relu'
FINAL_ACTIVATION = 'sigmoid'

METRICS = [
  tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=tf.float32),
  tf.keras.metrics.TruePositives(name='true_positives', dtype=tf.float32),
  tf.keras.metrics.FalsePositives(name='false_positives', dtype=tf.float32),
  tf.keras.metrics.TrueNegatives(name='true_negatives', dtype=tf.float32),
  tf.keras.metrics.FalseNegatives(name='false_negatives', dtype=tf.float32), 
  tf.keras.metrics.Precision(name='precision', dtype=tf.float32),
  tf.keras.metrics.Recall(name='recall', dtype=tf.float32),
  tf.keras.metrics.AUC(name='auc', dtype=tf.float32),
]

def generate_full_dataset():
  if not os.path.isdir('datasets'):
    os.makedirs('datasets')
  
  dirs = ['test', 'train', 'val']

  # prepare labels
  labels = ['type']
  for i in range(0, 96): # first dim
    for j in range(0, 96): # second dim
      for k in range(0, 96): # channel
        labels.append('{}x{}x{}'.format(i, j, k))

  # scan directorys
  for directory in dirs:
    dataset_path = 'datasets/dataset-{}.csv'.format(directory)
    try:
      f = open(dataset_path)
      f.close()
      print('Dataset already exists.')
    except FileNotFoundError:
      print('Generate full dataset with xray images... Please wait it can take a while.')
      with open(dataset_path, 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        # write first column labels
        writer.writerow(labels)

        i = 0
        for label in CLASS_NAMES:
          print('Scan {}/{}'.format(directory, label))
          with os.scandir('chest_xray/{}/{}'.format(directory, label)) as entries:
            for entry in entries:
              # load the image with opencv
              img = cv2.imread(entry.path)
              # resize it
              resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
              # 3D numpy array to 1D
              one_dim_img = resized_img.ravel()
              # write the row
              writer.writerow([label, *list(one_dim_img)])
              print('Row {} added.'.format(i))
              i = i + 1


def separate_pneumonia():
  """
  Little script to sperate types of pneumonia into 2 differents datasets, for train, test and val.
  """
  datas_types = ['train', 'test', 'val']
  for dtype in datas_types:
    # generate VIRUS and BACTERIA dirs
    os.mkdir('chest_xray/{}/VIRUS'.format(dtype), 0o755)
    os.mkdir('chest_xray/{}/BACTERIA'.format(dtype), 0o755)
    with os.scandir('chest_xray/{}/PNEUMONIA'.format(dtype)) as entries:
      for entry in entries:
        if entry.name.find('virus') > -1: # this is a virus
          print('Move virus img.')
          shutil.copy2(entry.path, 'chest_xray/{}/VIRUS/'.format(dtype))
        else:
          print('Move bacteria img.')
          shutil.copy2(entry.path, 'chest_xray/{}/BACTERIA/'.format(dtype))


def save_model_h5(model=None, model_name='vgg16'):
  """
  Save a TF Model into h5 format
  """
  model.save('saved_model/{}/model.h5'.format(MODEL_NAME))
  print("Model saved successfully.")

def save_model_tf(model=None, model_name='vgg16'):
  """
  Save a TF Model into SavedModel format
  """
  # Reset metrics before saving so that loaded model has same state,
  # since metric states are not preserved by Model.save_weights
  model.reset_metrics()

  model.save('saved_model/{}/model'.format(MODEL_NAME), save_format='tf')
  print("Model saved successfully.")

def get_callbacks():
  """
  Define the callbacks for the ML model
  """
  return [
    # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10),
    tf.keras.callbacks.TensorBoard(os.path.join("logs/{}".format(MODEL_NAME), datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1)
  ]

def get_label(file_path):
  """
  Get the label of a file - Can be NORMAL, VIRUS OR BACTERIE
  """
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  """
  Convert an image into a tensor with needed size
  """
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  """
  Process a file
  """
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def get_label_predicted(predictions):
  """
  Return the label for a given prediction array
  """
  preds = list(predictions)
  return CLASS_NAMES[preds.index(max(preds))]

def generate_results(history, model, results, predictions):
  """
  Function that generate an html page and open it in the browser
  - history is the history for the model train steps
  - results is the metrics results get from the train step
  - predictions is the predictive result gotten for the test datas
  """
  folder_results_name = 'results/{}_{}'.format(MODEL_NAME, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  os.mkdir(folder_results_name, 0o755)

  # Generate the graph for accuracy, loss, val_accuracy, val_loss
  history_dict = history.history

  acc = history_dict['accuracy']
  val_acc = history_dict['val_accuracy']
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']

  epochs_range = range(NB_EPOCHS)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.yscale('log')

  # save figure
  plt.savefig('{}/accuracy_and_loss.png'.format(folder_results_name))
	# close matplotlib
  plt.close()

  with open('{}/results.txt'.format(folder_results_name), 'w+', newline='') as f:
    # Write metrics for testing purpose
    f.write('Normal, Virus or Bacteria {} trained model\n'.format(MODEL_NAME))
    if MODEL_NAME == 'vgg16':
      f.write('Optimizer : {}, Hidden activation : {}, Final activation : {}\n'.format(OPTIMIZER, HIDDEN_ACTIVATION, FINAL_ACTIVATION))
    elif MODEL_NAME == 'resnet18' or MODEL_NAME == 'resnet34':
      f.write('Optimizer : {}, Final activation : {}\n'.format(OPTIMIZER, FINAL_ACTIVATION))
    f.write('--------------------------------------------------------------\n')
    f.write('\n')
    f.write('METRICS :\n')
    for name, value in zip(model.metrics_names, results):
      f.write(f'{name} : {value}\n')
    f.write('\n')
    # Write predictions
    f.write('PREDICTIONS :\n')
    for predict in predictions:
      f.write('Predictions : {}, Result : {}\n'.format(str(predict), get_label_predicted(predict)))

  
def main():
  """
  Main function
  """
  modelLoaded = False
  model = None
  # if the model already exist, load it.
  if (os.path.isfile('saved_model/{}/model.h5'.format(MODEL_NAME))):
    model = tf.keras.models.load_model('saved_model/{}/model.h5'.format(MODEL_NAME))
    modelLoaded = True
    print('Model successfully loaded.')
  else:
    # Get the model
    if MODEL_NAME == 'vgg16':
      model = get_model_vgg(
        model=MODEL_NAME,
        nodes=16,
        optimizer=OPTIMIZER,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        hidden_activation=HIDDEN_ACTIVATION,
        final_activation=FINAL_ACTIVATION,
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=tf.float32)]
      )
    elif MODEL_NAME == 'resnet18' or MODEL_NAME == 'resnet34':
      model = get_model_resnet(
        model=MODEL_NAME,
        optimizer=OPTIMIZER,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        final_activation=FINAL_ACTIVATION,
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=tf.float32)]
      )

  # To get the nb of steps and how many images we got
  nb_normal_tr = len(os.listdir('{}/NORMAL'.format(TRAIN_DATA_PATH)))
  nb_bacteria_tr = len(os.listdir('{}/BACTERIA'.format(TRAIN_DATA_PATH)))
  nb_virus_tr = len(os.listdir('{}/VIRUS'.format(TRAIN_DATA_PATH)))
  nb_normal_val = len(os.listdir('{}/NORMAL'.format(VAL_DATA_PATH)))
  nb_bacteria_val = len(os.listdir('{}/BACTERIA'.format(VAL_DATA_PATH)))
  nb_virus_val = len(os.listdir('{}/VIRUS'.format(VAL_DATA_PATH)))
  total_train = nb_normal_tr + nb_bacteria_tr + nb_virus_tr
  total_val = nb_normal_val + nb_bacteria_val + nb_virus_val

  # Our datas generators
  train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5
  ) # Generator for our training data
  validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
  test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

  train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    directory=TRAIN_DATA_PATH,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical'
  )
  val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    directory=VAL_DATA_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical'
  )
  test_data_gen = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    directory=TEST_DATA_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical'
  )

  # Train the model
  history = model.fit(
    train_data_gen,
    callbacks=get_callbacks(),
    steps_per_epoch=total_train // BATCH_SIZE,
    epochs=NB_EPOCHS,
    validation_data=val_data_gen,
    validation_steps=total_val // BATCH_SIZE
  )

  model.summary()

  # Use a testing model to display metrics
  testing_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
  testing_model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=METRICS
  )

  # Evalutate the model with test datas
  results = testing_model.evaluate(test_data_gen, verbose=0)

  # Predict the test datas
  predictions = testing_model.predict(test_data_gen)

  generate_results(history, testing_model, results, predictions)
  # save_model_h5(testing_model, 'resnet_18')

if __name__ == "__main__":
  # generate_full_dataset()
  main()
  exit()