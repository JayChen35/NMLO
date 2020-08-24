import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from dataparser import get_training_data, split

# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

X, y, class_names = get_training_data(False, True)
train_images, train_labels, test_images, test_labels = split(X, y, 0.1)

print(train_images.shape, train_labels.shape)
# print(1/0)

# Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0


model = models.Sequential()
model.add(layers.Conv2D(100, (3, 3), activation='relu', input_shape=train_images[0].shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(33, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    "checkpoint", monitor='val_loss', verbose=1, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch'
)

history = model.fit(train_images, train_labels, epochs=1000, 
                    validation_data=(test_images, test_labels),
                    callbacks=[checkpoint_callback])

print("Train accuracy:", history.history['sparse_categorical_accuracy'])
print("Validation accuracy:", history.history['val_accuracy'])

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Test accuracy:", test_acc)

model.save('model4')
