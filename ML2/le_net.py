import os
import warnings

import numpy as np
import tensorflow.keras as keras
import tqdm
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Dropout,
                                     Flatten, MaxPooling2D)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import History
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
FOLDER = "./MWI-Dataset/"
HAZE = "HAZE"
SUNNY = "SUNNY"
SNOWY = "SNOWY"
RAINY = "RAINY"
paths = [HAZE, SUNNY, SNOWY, RAINY]
IMG_SIZE = 200
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


# LE_NET model
model = Sequential()
model.add(Conv2D(32, (6, 6), padding="same", activation='relu',
                 input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
# model.add(Dropout(0.2))
model.add(Conv2D(32, (6, 6), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), padding='same'))
# model.add(Dropout(0.2))
model.add(Conv2D(32, (6, 6), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer='adam', metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    workers=100,
    steps_per_epoch=63,  # 2000 images over 32 batch size
    validation_data=valid_generator,
    epochs=20)

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
