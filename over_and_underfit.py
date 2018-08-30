### It may takes a long time and consumes large amount of RAM to execute...
## Import libraries
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# helper libraries
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

## Show TensorFlow version to make sure that TensorFlow is successfully loaded
print("Using TensorFlow", tf.__version__)

## Load dataset
# It uses the same dataset as text_classification.py
NUM_WORDS = 10000
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

## Create an all-zero matrix of shape (len(sequences), dimension)
def multi_hot_sequences(sequences, dimension):
    # Multi-hot-encoding: turn datasets into vectors of 0s and 1s
    results = np.zeros((len(sequences), dimension))

    for i, word_indices in enumerate(sequences):
        # set specific indices of results[i] to 1s
        results[i, word_indices] = 1.0

    return results

train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

plt.plot(train_data[0])
plt.savefig('output_images/t4_encoded_first_data.png')

## Demonstrate Overfitting
# Create a model
def build_model(size):
    model = keras.Sequential([
        keras.layers.Dense(size, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(size, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy']
    )

    return model

def fit_model(model):
    model.fit(
        train_data,
        train_labels,
        epochs=20,
        batch_size=512,
        validation_data=(test_data, test_labels),
        verbose=2
    )

# Create a baseline model
# >>> baseline_model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 16)                160016
# _________________________________________________________________
# dense_1 (Dense)              (None, 16)                272
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 17
# =================================================================
# Total params: 160,305
# Trainable params: 160,305
# Non-trainable params: 0
# _________________________________________________________________
baseline_model = build_model(16)
baseline_history = fit_model(baseline_model)

# Create a smaller model
smaller_model = build_model(4)
smaller_history = fit_model(smaller_model)

# Create a bigger model
bigger_model = build_model(512)
bigger_history = fit_model(bigger_model)

## Plot the training and validation loss
def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))

    # Training loss: solid line
    # Validation loss: dashed (--) line, **lower is better**
    for name, history in histories:
        val = plt.plot(
            # Traceback (most recent call last):
            # File "over_and_underfit.py", line 118, in <module>
            #     ('smaller', smaller_history),
            # File "over_and_underfit.py", line 96, in plot_history
            #     val = plt.plot(
            # AttributeError: 'NoneType' object has no attribute 'epoch'
            history.epoch,
            history.history['val_'+key],
            '--',
            label=name.title()+' Val'
        )
        plt.plot(
            history.epoch,
            history.history[key],
            color=val[0].get_color(),
            label=name.title()+' Train'
        )

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])

# The larger the network becomes, the faster overfitting begins
plot_history([
    ('baseline', baseline_history),
    ('smaller', smaller_history),
    ('bigger', bigger_history)
])
plt.savefig('output_images/t4_loss.png')

### Strategies
## Add weight regularization
def build_regularized_model(size):
    model = keras.Sequential([
        # L1 regularization: use absolute value of the weight
        # L2 regularization (weight decay): use squared value of the weight
        keras.layers.Dense(size, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(size, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy']
    )

    return model

l2_model = build_regularized_model(16)
l2_history = fit_model(l2_model)
plot_history([
    ('baseline', baseline_history),
    ('l2', l2_history),
])
plt.savefig('output_images/t4_loss_l2.png')

## Add dropout
def build_dropout_model(size):
    model = keras.Sequential([
        keras.layers.Dense(size, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        # randomly drop out layers
        keras.layers.Dropout(0.5),
        keras.layers.Dense(size, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy']
    )

    return model

dpt_model = build_dropout_model(16)
dpt_history = fit_model(dpt_model)
plot_history([
    ('baseline', baseline_history),
    ('dropout', dpt_history),
])
plt.savefig('output_images/t4_loss_dpt.png')
