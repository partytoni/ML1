from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import cv2
import os
import numpy as np

img_width, img_height = 150, 150
train_data_dir = "MWI-Dataset/"  # 2800 images folder
validation_data_dir = "MWI-Dataset-test/"  # 400 images folder
batch_size = 32
HAZE="HAZE"
SUNNY="SUNNY"
RAINY="RAINY"
SNOWY="SNOWY"
paths = [HAZE, SUNNY, RAINY, SNOWY]
STOP_COUNTER = 3

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)


def augment(folder, out_path):
    for classe in paths:
        extended_path = folder+classe+"/"
        listdir = os.listdir(extended_path)
        total = len(listdir)
        counter = 0
        for img_path in listdir:
            full_path = extended_path+img_path
            img = cv2.imread(full_path)
            img = cv2.resize(img, (img_height, img_width))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)

            print(classe,":",counter, "/", total)
            counter += 1

            in_counter = 0
            for image in train_datagen.flow(img,
                                            save_prefix="aug-",
                                            save_to_dir=out_path+classe,
                                            save_format="jpeg",
                                            batch_size=1):
                in_counter += 1

                if in_counter >= STOP_COUNTER:
                    break
                pass


augment(train_data_dir, "train-aug\\")
augment(validation_data_dir, "test-aug\\")