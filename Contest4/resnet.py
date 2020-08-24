import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# tf.config.set_per_process_memory_fraction(0.75)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# resnet_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

NUM_CLASSES = len(os.listdir("nmlo-contest-4/train/train/"))

print("Number of classes:", NUM_CLASSES)

# Still not talking about our train/test data or any pre-processing.
model = Sequential()

resnet_model = ResNet50(include_top=False, pooling='avg', weights='imagenet', input_shape=(244, 244, 3))

model.add(resnet_model)
# model.add(Dense(128, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))
# model.layers[0].trainable = False

for layer in resnet_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False

model.summary()

sgd = optimizers.SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])

image_size = (100, 100)
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.1)

# flow_From_directory generates batches of augmented data (where augmentation can be color conversion, etc)
# Both train & valid folders must have NUM_CLASSES sub-folders
train_generator = data_generator.flow_from_directory(
        'nmlo-contest-4/train/train',
        target_size=image_size,
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=120)

validation_generator = data_generator.flow_from_directory(
        'nmlo-contest-4/train/train',
        target_size=image_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=120)

print("Training stuff:", len(train_generator), len(validation_generator))
# print(next(train_generator))
# print(1/0)

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience=3)
cb_checkpointer = ModelCheckpoint(filepath = 'bestest.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')

fit_history = model.fit_generator(
        train_generator,
        # steps_per_epoch=50,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=10,
        callbacks=[cb_checkpointer, cb_early_stopper]
)

print(fit_history)

