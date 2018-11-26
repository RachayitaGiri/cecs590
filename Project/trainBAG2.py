import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras import applications

# dimensions of our images.
img_width, img_height = 96, 96

train_data_dir = 'CALTrain'
validation_data_dir = 'CALTrain'
nb_train_samples = 20
nb_validation_samples = 20
epochs = 50
batch_size = 100

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)




def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                    activation='relu',
                    input_shape=(80,80,3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2000, activation='softmax'))
    model.add(Dense(102, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def getAlexNet():
    #1
    model = Sequential()
    # model.add(Conv2D(96, (11, 11), input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # #2
    # model.add(Conv2D(256, (5, 5)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # #3
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # #4
    # model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #5
    model.add(Conv2D(1024, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #6
    model.add(Flatten())
    model.add(Dense(3072))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #7
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #8
    model.add(Dense(102))
    model.add(Activation('sigmoid'))
    return model


model = getAlexNet() 
model.summary()

# model = baseline_model()
# model.summary()



model.compile(loss='binary_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator()

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()
datagen = ImageDataGenerator()
test_datagen = datagen

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples ,
    epochs=epochs,
    validation_data=None,
    validation_steps=nb_validation_samples)

# save_bottlebeck_features()
# train_top_model()