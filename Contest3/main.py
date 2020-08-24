import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop

from dataparser import get_training_data


X_train, y_train = get_training_data()
print(X_train.shape, y_train.shape)

model = Sequential()

# The Input Layer :
model.add(Dense(64, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))
# The Hidden Layers :
model.add(Dropout(0.1))
# model.add(Dense(64, kernel_initializer='normal',activation='relu'))
# The Output Layer:
model.add(Dense(1, kernel_initializer='normal', activation='linear'))

# # Load the network :
# weights_file = 'weights/FinalWeights.h5' # choose the best checkpoint 
# model.load_weights(weights_file) # load it

# Compile the network :
opt = RMSprop(learning_rate=0.005)
model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])
model.summary()

checkpoint_name = 'weights3/Weights-{epoch:03d}--{val_mean_absolute_error:.2f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_mean_absolute_error', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks_list,
    verbose=1
)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("weights3/FinalWeights.h5")

