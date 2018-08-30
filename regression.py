## Import libraries
from __future__ import absolute_import, division, print_function
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
# The Boston Housing Prices dataset
boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# shuffle the dataset
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
# >>> print(train_labels[0:10])
# [32.  27.5 32.  23.1 50.  20.6 22.6 36.2 21.8 19.5]
train_labels = train_labels[order]

# 404 examples, 13 features
print('Training set: {} examples, {} features'.format(train_data.shape[0], train_data.shape[1]))
# 102 examples, 13 features
print('Testing set: {} examples, {} features'.format(test_data.shape[0], test_data.shape[1]))

## Display dataset as a table
# import pandas as pd
# column_names = [
#     # Per capita crime rate
#     'CRIM',
#     # The proportion of residential land zoned for lots over 25,000 square feet
#     'ZN',
#     # The proportion of non-retail business acres per town
#     'INDUS',
#     # Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#     'CHAS',
#     # Nitric oxides concentration (parts per 10 million)
#     'NOX',
#     # The average number of rooms per dwelling
#     'RM',
#     # The proportion of owner-occupied units built before 1940
#     'AGE',
#     # Weighted distances to five Boston employment centers
#     'DIS',
#     # Index of accessibility to radial highways
#     'RAD',
#     # Full-value property-tax rate per $10,000
#     'TAX',
#     # Pupil-teacher ratio by town
#     'PTRATIO',
#     # 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town
#     'B',
#     # Percentage lower status of the population
#     'LSTAT'
# ]
# df = pd.DataFrame(train_data, columns=column_names)
# >>> df.head()
#       CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS   RAD    TAX  PTRATIO       B  LSTAT
# 0  0.07875  45.0   3.44   0.0  0.437  6.782  41.1  3.7886   5.0  398.0     15.2  393.87   6.68
# 1  4.55587   0.0  18.10   0.0  0.718  3.561  87.9  1.6132  24.0  666.0     20.2  354.70   7.12
# 2  0.09604  40.0   6.41   0.0  0.447  6.854  42.8  4.2673   4.0  254.0     17.6  396.90   2.98
# 3  0.01870  85.0   4.15   0.0  0.429  6.516  27.7  8.5353   4.0  351.0     17.9  392.43   6.36
# 4  0.52693   0.0   6.20   0.0  0.504  8.725  83.0  2.8944   8.0  307.0     17.4  382.00   4.63

## Normalize features
# Test data is *not* used when calculating the mean and std
mean = train_data.mean(axis=0)
# std: Standard Deviation (SD)
std = train_data.std(axis=0)

# >>> print(train_data[0])
# [-0.39725269  1.41205707 -1.12664623 -0.25683275 -1.027385    0.72635358
#  -1.00016413  0.02383449 -0.51114231 -0.04753316 -1.49067405  0.41584124
#  -0.83648691]
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

### Helper Methods
## Build a new model
def build_model():
    model = keras.Sequential([
        # train_data.shape[1]: features
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    model.compile(
        # Mean Squared Error (MSE)
        loss='mse',
        optimizer=tf.train.RMSPropOptimizer(0.001),
        # Mean Absolute Error (MAE)
        metrics=['mae']
    )

    return model

## Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')

## Visualize train and validation loss
def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error [1000$]')

    # >>> history.history.keys()
    # dict_keys(['val_loss', 'val_mean_absolute_error', 'loss', 'mean_absolute_error'])
    plt.plot(
        history.epoch,
        np.array(history.history['mean_absolute_error']),
        label='Train Loss'
    )

    plt.plot(
        history.epoch,
        np.array(history.history['val_mean_absolute_error']),
        label='Validation Loss'
    )

    plt.legend()
    plt.ylim([0, 5])

## Create the model
# >>> model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 64)                896
# _________________________________________________________________
# dense_1 (Dense)              (None, 64)                4160
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 65
# =================================================================
# Total params: 5,121
# Trainable params: 5,121
# Non-trainable params: 0
model = build_model()

## Train the model
epochs = 500

# Store training stats
history = model.fit(
    train_data,
    train_labels,
    epochs=epochs,
    validation_split=0.2,
    verbose=0,
    callbacks=[PrintDot()]
)

plot_history(history)
plt.savefig('output_images/t3_loss.png')

model = build_model()
history = model.fit(
    train_data,
    train_labels,
    epochs=epochs,
    validation_split=0.2,
    verbose=0,
    # automatically stop training when the validation score doesn't improve
    callbacks=[
        PrintDot(),
        # The patience parameter is the amount of epochs to check for improvement
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    ]
)
plot_history(history)
plt.savefig('output_images/t3_loss_es.png')

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
# Testing set Mean Absolute Error: $2548.64
print("Testing set Mean Absolute Error: ${:7.2f}".format(mae*1000))

## Predict
test_predictions = model.predict(test_data).flatten()

# draw result
plt.clf()
plt.scatter(test_labels, test_predictions)
# disable axis label
# (-24.950000000000003, 523.95, -1.598052716085393, 51.83531789762637)
plt.axis('equal')
# (-24.950000000000003, 523.95)
plt.xlim(plt.xlim())
# (-1.598052716085393, 51.83531789762637)
plt.ylim(plt.ylim())
plt.plot([-100, 100], [-100, 100])
plt.savefig('output_images/t3_result.png')

# draw error
plt.clf()
error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction Error [1000$]")
plt.ylabel("Count")
plt.savefig('output_images/t3_error.png')
