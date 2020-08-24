import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, AveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf


NUM_CLASSES = 33

resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

headModel = resnet_model.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(NUM_CLASSES, activation="softmax")(headModel)

model = Model(inputs=resnet_model.input, outputs=headModel)

for layer in resnet_model.layers:
    layer.trainable = False

# sgd = optimizers.SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
adam = optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

image_size = (224, 224)
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.1)

# flow_From_directory generates batches of augmented data (where augmentation can be color conversion, etc)
# Both train & valid folders must have NUM_CLASSES sub-folders
train_generator = data_generator.flow_from_directory(
        'nmlo-contest-4/resized',
        target_size=image_size,
        batch_size=16,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=80)

validation_generator = data_generator.flow_from_directory(
        'nmlo-contest-4/resized',
        target_size=image_size,
        batch_size=16,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=80)

print("Training stuff:", len(train_generator), len(validation_generator))

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


