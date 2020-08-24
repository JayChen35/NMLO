import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

from dataparser import get_testing_data, submit

test = get_testing_data()


def submit_nn():
    model = Sequential()

    # The Input Layer :
    model.add(Dense(64, kernel_initializer='normal',input_dim = test.shape[1], activation='relu'))
    # The Hidden Layers :
    model.add(Dropout(0.1))
    # model.add(Dropout(0.2))
    # model.add(Dense(64, kernel_initializer='normal',activation='relu'))
    # The Output Layer:
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))


    # Load wights file of the best model :
    weights_file = 'weights3/Weights-144--343.17.hdf5' # choose the best checkpoint 
    model.load_weights(weights_file) # load it
    model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['mean_absolute_error'])

    predictions = model.predict(test)
    predictions = [pred[0] for pred in predictions]
    submit(predictions, "submission.csv")

submit_nn()
# submit_xgb()
# submit_forest()
