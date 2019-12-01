import os
import warnings

import numpy as np
import tensorflow.keras as keras
import tqdm
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Dropout,
                                     Flatten, MaxPooling2D, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import History
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches


IMG_SIZE = 250
warnings.filterwarnings("ignore")


# FIRST MODEL TEST

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=[0.8, 1.2],
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True)

train_generator = datagen.flow_from_directory(
    directory=r"./MWI-Dataset/",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=64,
    class_mode="categorical",
    shuffle=True
)

valid_generator = datagen.flow_from_directory(
    directory=r"./MWI-Dataset-test/",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)
valid_generator2 = datagen.flow_from_directory(
    directory=r"./MWI-Dataset-test2/",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)
valid_generator3 = datagen.flow_from_directory(
    directory=r"./MWI-Dataset-test3/",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)


# ALEX_NET model
model = Sequential()

# 1st layer
model.add(Conv2D(96, (11, 11), strides=(4, 4), padding="valid", activation='relu',
                 input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='valid'))
model.add(BatchNormalization())

# 2nd layer
model.add(Conv2D(256, (5, 5), strides=(1, 1),
                 padding='valid', activation='relu'))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='valid'))
model.add(BatchNormalization())

# 3rd layer
model.add(Conv2D(384, (3, 3), strides=(1, 1),
                 padding='valid', activation='relu'))
model.add(BatchNormalization())

# 4th layer
model.add(Conv2D(384, (3, 3), strides=(1, 1),
                 padding='valid', activation='relu'))
model.add(BatchNormalization())

# 5th layer
model.add(Conv2D(256, (3, 3), strides=(1, 1),
                 padding='valid', activation='relu'))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='valid'))
model.add(BatchNormalization())

model.add(Flatten())


#dense layers
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

#output layer
model.add(Dense(4, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer='nadam', metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    workers=100,
    steps_per_epoch=63,  # 2000 images over 32 batch size
    validation_data=valid_generator,
    epochs=10)

scores = model.evaluate_generator(valid_generator)
print("\n\n1 - Accuracy 1st test: "+str(scores[1]))
"""scores = model.evaluate_generator(valid_generator2)
print("\n\n1 - Accuracy 2nd test: "+str(scores[1]))
scores = model.evaluate_generator(valid_generator3)
print("\n\n1 - Accuracy 3rd test: "+str(scores[1])) """

loss1 = history.history['loss']
val_loss1 = history.history['val_loss']

plt.plot(loss1, color="red")
plt.plot(val_loss1, color="blue")
red_patch = mpatches.Patch(color="red", label="loss")
blue_patch = mpatches.Patch(color="blue", label="val_loss")
plt.legend(handles=[red_patch, blue_patch])
plt.show()

# ALEX_NET MODEL
