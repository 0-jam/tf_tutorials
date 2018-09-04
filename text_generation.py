### It may takes a long time to execute
## Import libraries for simulation
import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import re
import random
import time
import unidecode

## Show TensorFlow version to make sure that TensorFlow is successfully loaded
print("Using TensorFlow", tf.__version__)

## Load dataset
file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/yashkatariya/shakespeare.txt')
text = unidecode.unidecode(open(file).read())
print("Length of text (the number of characters):", len(text))

# all the unique characters in the file
unique = sorted(set(text))
# creating a mapping from unique characters to indices
char2idx = {char:index for index, char in enumerate(unique)}
idx2char = {index:char for index, char in enumerate(unique)}

# Note: 1/2 embedding_dim and units to shorten running time
# set the maximum length of sentence
max_length = 100
vocab_size = len(unique)
embedding_dim = 128
# embedding_dim = 256
# number of RNN (Recursive Neural Network) units
units = 512
# units = 1024
batch_size = 64
# buffer size to shuffle our dataset
buffer_size = 10000

## Create the input and output tensors
input_text = []
target_text = []

for i in range(0, len(text) - max_length, max_length):
    input = text[i:i + max_length]
    target = text[i + 1:i + 1 + max_length]

    input_text.append([char2idx[j] for j in input])
    target_text.append([char2idx[k] for k in target])

print("Input text shape:", np.array(input_text).shape)
print("Target text shape:", np.array(target_text).shape)

## Create and shuffle batches
dataset = tf.data.Dataset.from_tensor_slices((input_text, target_text)).shuffle(buffer_size)
dataset = dataset.batch(batch_size, drop_remainder=True)

class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size):
        super(Model, self).__init__()
        self.units = units
        self.batch_sz = batch_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # check CUDA is able to use
        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(
                self.units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform'
            )
        else:
            self.gru = tf.keras.layers.GRU(
                self.units,
                return_sequences=True,
                return_state=True,
                recurrent_activation='sigmoid',
                recurrent_initializer='glorot_uniform'
            )

        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden):
        x = self.embedding(x)

        # output_shape = (batch_size, max_length, hidden_size)
        # states_shape = (batch_size, hidden_size)

        # states variable to preserve the state of the model
        # this will be used to pass at every step to the model while training
        output, states = self.gru(x, initial_state=hidden)

        # reshaping the output so that we can pass it to the Dense layer
        # after reshaping the shape is (batch_size * max_length, hidden)
        output = tf.reshape(output, (-1, output.shape[2]))

        # the dense layer will output predictions for every time_steps(max_length)
        # output shape after the dense layer is (batch_size * max_length, vocab_size)
        x = self.fc(output)

        return x, states

## Call the model
model = Model(vocab_size, embedding_dim, units, batch_size)
## Set the optimizer and the loss function
optimizer = tf.train.AdamOptimizer()

def loss_function(real, preds):
    # using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)

## Train the model
# Note: 1/30 epoch size to shorten running time
epochs = 5
# epochs = 30

for epoch in range(epochs):
    start = time.time()

    # initializing the hidden state at the start of every epoch
    hidden = model.reset_states()

    for (batch, (input, target)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            # feed the hidden state back into the model
            predictions, hidden = model(input, hidden)

            # reshape target to make loss function expect the target
            target = tf.reshape(target, (-1,))
            loss = loss_function(target, predictions)

        gradients = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(gradients, model.variables), global_step = tf.train.get_or_create_global_step())

        print("Epoch: {} / {}, Batch: {}, Loss: {:.4f}".format(epoch + 1, epochs, batch + 1, loss))

    print("Time taken for 1 epoch {} sec \n".format(time.time() - start))

## Predict trained model
gen_size = 1000
generated_text = ''
start_string = 'Q'
# convert start_string to number
input_eval = tf.expand_dims([char2idx[s] for s in start_string], 0)
# low temperatures results in more predictable text
# higher temperatures results in more surprising text
temperature = 1.0

# hidden state shape is (batch_size, units)
hidden = [tf.zeros((1, units))]

for i in range(gen_size):
    print("prediction {} / {}".format(i + 1, gen_size))
    predictions, hidden = model(input_eval, hidden)

    # use a multinomial distribution to predict the word returned by the model
    predictions = predictions / temperature
    predicted_id = tf.multinomial(tf.exp(predictions), num_samples = 1)[0][0].numpy()

    # pass the predicted word as the next input to the model along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)

    generated_text += idx2char[predicted_id]

print(start_string + generated_text)
