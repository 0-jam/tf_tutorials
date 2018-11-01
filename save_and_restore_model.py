## Import libraries
from __future__ import absolute_import, division, print_function
import os, pathlib
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

## Show TensorFlow version to make sure that TensorFlow is successfully loaded
print("Using TensorFlow", tf.__version__)

## Load dataset
# Use MNIST handwritten digits dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

## Define a model
# Returns a short sequential model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    return model

# Create a basic model instance
model = build_model()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_2 (Dense)              (None, 512)               401920
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 512)               0
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                5130
# =================================================================
# Total params: 407,050
# Trainable params: 407,050
# Non-trainable params: 0
# _________________________________________________________________
model.summary()

## Save checkpoints during training
# Automatically save checkpoints during at the end of training
# Checkpoint callback usage
checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create checkpoint callback
cp_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    verbose=1
)

model = build_model()
model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    # pass callback to training
    callbacks=[cp_callback]
)

# Create untrained model
model = build_model()
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# Load the weights from the checkpoint
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# Checkpoint callback options
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create checkpoint callback
cp_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    verbose=1,
    # Save weights every 5 epochs
    period=5
)

model = build_model()
model.fit(
    train_images,
    train_labels,
    epochs=50,
    validation_data=(test_images, test_labels),
    # pass callback to training
    callbacks=[cp_callback],
    verbose=0
)

# Sort the checkpoints by modification time.
checkpoints = sorted(
    pathlib.Path(checkpoint_dir).glob("*.index"),
    # mtime: last modificated time
    key=lambda cp: cp.stat().st_mtime
)
# Note: the default tensorflow format only saves the 5 most recent checkpoints.
# [
#     PosixPath('training_2/cp-0046.ckpt'),
#     PosixPath('training_2/cp-0047.ckpt'),
#     PosixPath('training_2/cp-0048.ckpt'),
#     PosixPath('training_2/cp-0049.ckpt'),
#     PosixPath('training_2/cp-0050.ckpt')
# ]
checkpoints = [cp.with_suffix('') for cp in checkpoints]
# -1: Last index of list
latest = str(checkpoints[-1])

# Create untrained model
model = build_model()
# Load the weights from the checkpoint
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
# Restored model, accuracy: 87.50%
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

## Manually save weights
my_checkpoint = './my_checkpoint/my_checkpoint'

# Save the weights
model.save_weights(my_checkpoint)

# Restore the weights
model = build_model()
model.load_weights(my_checkpoint)

loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

## Save the entire model
model = build_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file
model_path = 'my_checkpoint/my_model.h5'
model.save(model_path)

# Recreate the model from previous file
model2 = keras.models.load_model(model_path)

loss, acc = model2.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
