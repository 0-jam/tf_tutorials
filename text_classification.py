## Import libraries
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# helper libraries
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

## Show TensorFlow version
print("Using TensorFlow", tf.__version__)

## Load dataset
imdb = keras.datasets.imdb
# keep top 10000 most frequently occuring words in the training data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Training entries: 25000, labels: 25000
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# [1, 14, 22, ..., 178, 32]
print("First training data:", train_data[0])
# Movie reviews may be different lengths
# 218
print("Length of first training data:", len(train_data[0]))
# 189
print("Length of second training data:", len(train_data[1]))

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNKNOWN>"] = 2
word_index["<UNUSED>"] = 3

# A dictionary mapping integer indices to a word
# swap word_index key for value
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

## Convert the integers back to words
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

## Prepare the data
# convert (pad) each data to a same length array
# (max_length * num_reviews) shape tensor
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index['<PAD>'],
    padding='post',
    maxlen=256
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index['<PAD>'],
    padding='post',
    maxlen=256
)

# 256
print("Length of first training data (after padding):", len(train_data[0]))
# 256
print("Length of second training data (after padding):", len(train_data[1]))

## Build the model
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential([
    # word embedding layer
    # takes vector for each word-index
    keras.layers.Embedding(vocab_size, 16),
    # convert word vector to fixed length vector
    keras.layers.GlobalAveragePooling1D(),
    # pipe previous layer
    # fully-connected (densely connected) layer with 16 hidden units
    keras.layers.Dense(16, activation=tf.nn.relu),
    # sigmoid activation function (returns probability value)
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

print("Model summary:")
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, None, 16)          160000
# _________________________________________________________________
# global_average_pooling1d (Gl (None, 16)                0
# _________________________________________________________________
# dense_2 (Dense)              (None, 16)                272
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 17
# =================================================================
# Total params: 160,289
# Trainable params: 160,289
# Non-trainable params: 0
# _________________________________________________________________
model.summary()

## Compile the model
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

## Create a validation set to check the accuracy of the model
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

## Train the model
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

## Evaluate the model
results = model.evaluate(test_data, test_labels)
# results: [0.3048070820426941, 0.87608]
print("Loss: {}, accuracy: {} %".format(results[0], results[1]*100))

### Create a graph of accuracy and loss
history_dict = history.history
# >>> history_dict.keys()
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

epochs = range(1, len(history_dict['acc']) + 1)

## Loss graph
# 'bo': blue dot
plt.plot(epochs, history_dict['loss'], 'bo', label='Training loss')
# 'b': solid blue line
plt.plot(epochs, history_dict['val_loss'], 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('output_images/t2_loss.png')

# clear figure
plt.clf()

## Accuracy graph
plt.plot(epochs, history_dict['acc'], 'bo', label='Training accuracy')
plt.plot(epochs, history_dict['val_acc'], 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('output_images/t2_acc.png')
