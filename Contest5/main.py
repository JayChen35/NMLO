# https://www.tensorflow.org/hub/tutorials/tf2_text_classification

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
from dataparser import get_training_data, split, get_testing_data

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
# print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], 
#                                   batch_size=-1, as_supervised=True)

# train_examples, train_labels = tfds.as_numpy(train_data)
# test_examples, test_labels = tfds.as_numpy(test_data)

training_data, training_labels = get_training_data()
print(training_data.shape, training_labels.shape)
train_examples, train_labels, test_examples, test_labels = split(training_data, training_labels, 0.1)

# print(training_data.shape, training_labels.shape)

print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))
print(train_examples[:10])
print(train_labels[:10])

model_path = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
model_path2 = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1"
model_path3 = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"
model_path4 = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
hub_layer = hub.KerasLayer(model_path3, output_shape=[20], input_shape=[], 
                           dtype=tf.string, trainable=True)

model = tf.keras.Sequential()
model.add(hub_layer)
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# max_features = 6000
# embed_size = 128
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Embedding(max_features, embed_size))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences = True)))
# model.add(tf.keras.layers.GlobalMaxPool1D())
# model.add(tf.keras.layers.Dense(20, activation="relu"))
# model.add(tf.keras.layers.Dropout(0.05))
# model.add(tf.keras.layers.Dense(1, activation="sigmoid"))


model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

x_train, y_train, x_val, y_val = split(train_examples, train_labels, 0.1)

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience=3)
cb_checkpointer = ModelCheckpoint(filepath = 'bestmodel.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')

history = model.fit(x_train,
                    y_train,
                    epochs=50,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1, callbacks=[cb_early_stopper, cb_checkpointer])

model.load_weights("bestmodel.hdf5")


results = model.evaluate(test_examples, test_labels)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))

# model.save("rappop")

## SUBMISSION CODE

# testing_ids, testing_data = get_testing_data()
# pred = model.predict(testing_data)
# # print(pred)
# print(pred.shape)
# print(pred[:30])

# file = open("submission.csv", "w+")
# file.write("id,class\n")
# for i in range(len(pred)):
#     label = 1 if pred[i] > 0.0 else 0
#     out = str(testing_ids[i]) + "," + str(label) + "\n"
#     file.write(out)

# file.close()










# history_dict = history.history
# history_dict.keys()


# acc = history_dict['accuracy']
# val_acc = history_dict['val_accuracy']
# loss = history_dict['loss']
# val_loss = history_dict['val_loss']

# epochs = range(1, len(acc) + 1)

# # "bo" is for "blue dot"
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

# plt.clf()   # clear figure
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.show()
