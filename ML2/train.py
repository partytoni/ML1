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


FOLDER = "./MWI-Dataset/"
HAZE = "HAZE"
SUNNY = "SUNNY"
SNOWY = "SNOWY"
RAINY = "RAINY"
paths = [HAZE, SUNNY, SNOWY, RAINY]
IMG_SIZE = 300
warnings.filterwarnings("ignore")


class TestCallback():
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
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

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',
                 input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer='adam', metrics=['accuracy'])

model.fit_generator(
    train_generator,
    workers=100,
    epochs=20)
# validation_data=valid_generator)

scores = model.evaluate_generator(valid_generator)
print("\n\nAccuracy: "+str(scores[1]))
