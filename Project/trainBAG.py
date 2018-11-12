import keras
import os, sys

import PIL
from PIL import Image
from resizeimage import resizeimage

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.utils import np_utils

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator

def baseline_model():
	# create model
	model = Sequential([
        Convolution2D(32,3,3, input_shape=(224,224,3), kernel_initializer='normal', activation='relu'),
        MaxPooling2D(pool_size = (2,2)),
        Flatten(),
        Dense(output_dim = 128, activation = "relu"),
        Dense(output_dim = 1, activation = "sigmoid")])
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

trainSet = ImageDataGenerator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range = 0.2,
                                horizontal_flip=True)
trainSetData = trainSet.flow_from_directory(
        'trainSet',
        target_size=(224, 224),
        batch_size=5,
        class_mode='binary')


testSetData = trainSet.flow_from_directory(
        'trainSet',
        target_size=(224, 224),
        batch_size=10,
        class_mode='binary')


print(len(trainSetData))
    
# build the model
model = baseline_model()
# Fit the model
model.fit_generator(trainSetData, steps_per_epoch=500, epochs=10, validation_data=testSetData, validation_steps = 100)