import keras
import keras.utils
import sklearn
from sklearn.model_selection import cross_validate
import sklearn.metrics
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import utils as np_utils
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, MaxPool2D, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

DATASET_PATH = '../input/15-scene/15-Scene/'

one_hot_lookup = np.eye(15)  # 15 classes

dataset_x = []
dataset_y = []

for category in sorted(os.listdir(DATASET_PATH)):
    print('loading category: ' + str(int(category)))
    for fname in os.listdir(DATASET_PATH + category):
        img = cv2.imread(DATASET_PATH + category + '/' + fname, 2)
        img = cv2.resize(img, (224, 224))
        dataset_x.append(np.reshape(img, [224, 224, 1]))
        dataset_y.append(np.reshape(one_hot_lookup[int(category)], [15]))

dataset_x = np.array(dataset_x)
dataset_y = np.array(dataset_y)

"""shuffle dataset"""
p = np.random.permutation(len(dataset_x))
dataset_x = dataset_x[p]
dataset_y = dataset_y[p]

X_test = dataset_x[:int(len(dataset_x) / 10)]
Y_test = dataset_y[:int(len(dataset_x) / 10)]
X_train = dataset_x[int(len(dataset_x) / 10):]
Y_train = dataset_y[int(len(dataset_x) / 10):]