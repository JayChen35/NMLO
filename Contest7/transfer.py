import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential

tf.compat.v1.disable_eager_execution()

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# Import the Universal Sentence Encoder's TF Hub module
print("Fetching hub module")
embed = hub.Module(module_url)
hub_layer = hub.KerasLayer(module_url, output_shape=[40], input_shape=[], 
                           dtype=tf.string, trainable=True)
print("Fetched it")

embed = Sequential()
embed.add(hub_layer)

# Compute a representation for each message, showing various lengths supported.
messages = ["That band rocks!", "That song is really cool."]

# with tf.Session() as session:
#   session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#   message_embeddings = session.run(embed(messages))

message_embeddings = embed(messages)
print(message_embeddings)
