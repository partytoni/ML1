from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

model = load_model("vgg16_1_acc0.90.h5")
model_loss = load_model("vgg16_1_bestloss.h5")
img_height, img_width = 150,150
#validation_data_dir = "MWI-Dataset-test"
validation_data_dir = "Weather_Dataset"


test_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    shuffle=False,
    class_mode="categorical")

""" scores = model.evaluate_generator(validation_generator, verbose=1)
print("With best accuracy model:", scores)
scores_loss = model_loss.evaluate_generator(validation_generator, verbose=1)
print("With lowest val_loss model:",scores_loss)
 """

preds = model.predict_generator(validation_generator, verbose=1)

Ypred = np.argmax(preds, axis=1)
Ytest = validation_generator.classes  # shuffle=False in test_generator

print(classification_report(Ytest, Ypred, labels=None,
                            target_names=["Haze", "Rainy", "Snowy", "Sunny"], digits=3))
