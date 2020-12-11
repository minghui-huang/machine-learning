# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import os
# from urllib.request import urlopen,urlretrieve
# from PIL import Image
# from tqdm import tqdm_notebook
# from sklearn.utils import shuffle
import cv2
# from keras.models import load_model
# from sklearn.datasets import load_files
from keras.utils import np_utils
# from glob import glob
from keras import applications
# from keras.preprocessing.image import ImageDataGenerator
# from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
# from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from keras.optimizers import SGD, Adam

train = np.load("Images_32.npy")
label = np.load("Labels_32.npy")
resize_set = []

# gray = []
# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
#
# for img in train:
#     gray.append(rgb2gray(img))


height = 32

for origin_sz_pic in train:
    narw_size = cv2.resize(origin_sz_pic, dsize=(height, height), interpolation=cv2.INTER_CUBIC)
    resize_set.append(narw_size)
print("resized picture set")

resize_set = np.asarray(resize_set)

# img_train = resize_set[:2000]
# label_train = label[:2000]
# img_test = resize_set[2000:2850]
# label_test = label[2000:2850]

img_train = resize_set[:9000]
label_train = label[:9000]
img_test = resize_set[9000:]
label_test = label[9000:]

# Flattening the images from the 28x28 pixels to 1D 787 pixels
X_train = img_train.reshape(img_train.shape[0], height,height,3)
X_test = img_test.reshape(img_test.shape[0], height, height,3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

n_classes = 5
print("Shape before one-hot encoding: ", label_train.shape)
Y_train = np_utils.to_categorical(label_train, n_classes)
Y_test = np_utils.to_categorical(label_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


#from tensorflow.keras.applications.resnet50 import ResNet50

base_model = applications.resnet50.ResNet50(weights= 'imagenet', include_top=False, input_shape= (height, height, 3))
#base_model = applications.resnet50.ResNet50(weights= 'imagenet', include_top=True,classes=1000)
# base_model = ResNet50(weights='imagenet')


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)

predictions = Dense(5, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.0005)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=40, batch_size = 100)
# validation_data=(X_test, Y_test)

#sample_weights = np.ones(5)
#learning_phase = 1  # 1 means "training"
#ins = [X_test, Y_test, sample_weights, learning_phase]
#print(ins)
#model.evaluate(X_test, Y_test, sample_weight= sample_weights)

preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
predictions = model.predict
print(predictions)

# result = []
#
# for ary in predictions:
#     ary = ary.reshape(-1)
#     max = np.argmax(ary)
#     result.append(max)
# print(result)
