import tensorflow_datasets as tfds
train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], 
                                  batch_size=-1, as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

print(train_examples.shape, train_labels.shape)
print(train_examples.dtype, train_labels.dtype)