from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from dataparser import get_testing_data

NUM_CLASSES = 33

model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.layers[0].trainable = False
model.summary()

model.load_weights("best.hdf5")

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = data_generator.flow_from_directory(
    directory = 'nmlo-contest-4/test/',
    target_size = (100, 100),
    batch_size = 1,
    class_mode = None,
    shuffle = False,
    seed = 123
)

test_generator.reset()
pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
# pred = np.argmax(pred, axis=1)

testing_imgs, testing_ids, class_labels = get_testing_data()
file = open("submission.csv", "w+")
file.write("id,label\n")
for i in range(len(pred)):
    # idx = pred[i]
    idx = np.argmax(pred[i])
    out = str(testing_ids[i]) + "," + class_labels[idx] + "\n"
    file.write(out)

file.close()

