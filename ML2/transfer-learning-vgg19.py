import tensorflow
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import warnings
from tensorflow.keras.callbacks import History
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt


###################STATIC VALUES###################
img_width, img_height = 150, 150
# train_data_dir = "MWI-Dataset"  #2800 images folder
train_data_dir = "aug-set"  # 2800 images folder
#train_data_dir = "Weather_Dataset"  # 2800 images folder
validation_data_dir = "MWI-Dataset-test" #400 images folder
#validation_data_dir = "test-aug"  # 400 images folder
batch_size = 64
epochs = 10
NON_TRAINABLE_LAYERS = 10  # how many non trainable layers we choose


warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module=r'.*TiffImagePlugin'
)
###################################################


##########################GENERATORS##############################
# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

empty_train_datagen = ImageDataGenerator()
empty_test_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode="categorical")
##################################################################




##############################FUNCTIONS###########################
def get_model():

    model = applications.VGG19(
        weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

    # we set the first NON_TRAINABLE_LAYERS to non trainable, then the others to trainable
    for layer in model.layers[:NON_TRAINABLE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[NON_TRAINABLE_LAYERS:]:
        layer.trainable = True

    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(4, activation="softmax")(x)

    # creating the final model
    model_final = tensorflow.keras.Model(
        inputs=model.input, outputs=predictions)

    # compile the model
    model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(
        lr=0.0001), metrics=["accuracy"])

    return model_final


def plot(history):
    # we plot the loss-val_loss curves
    loss1 = history.history['loss']
    val_loss1 = history.history['val_loss']
    plt.plot(loss1, color="red")
    plt.plot(val_loss1, color="blue")
    axes = plt.gca()
    axes.set_ylim([0, 1])
    red_patch = mpatches.Patch(color="red", label="loss")
    blue_patch = mpatches.Patch(color="blue", label="val_loss")
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()


def get_callbacks():
    # Save the model according to the conditions
    checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)
    checkpoint_loss = ModelCheckpoint("vgg16_1_bestloss.h5", monitor='val_loss', verbose=1,
                                      save_best_only=True, save_weights_only=False, mode='min', period=1)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0,
                          patience=10, verbose=1, mode='auto')

    return [checkpoint_loss, checkpoint, early]


def main():
    # to train a new model uncomment this
    model = get_model()

    # to start from a previously trained model uncomment this
    #model = load_model("vgg16_1.h5")

    # Train the model
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=get_callbacks())

    plot(history)


main()
