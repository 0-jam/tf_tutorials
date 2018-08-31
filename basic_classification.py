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
# Fashion MNIST dataset: 70000 grayscale images of clothes in 10 categories
fashion_mnist = keras.datasets.fashion_mnist
# associate image with label
# train_*: training set
# test_*: test set
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# class name associate with the label
class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

### format of each datasets
## number and resolution of images in the training set
# >>> train_images.shape
# (60000, 28, 28)
## number of labels in the training set
# >>> len(train_labels)
# 60000
## format of labels (in this case, each labels is an integer between 0 and 9)
# >>> train_labels
# array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
# >>> test_images.shape
# (10000, 28, 28)
## number and resolution of images in the test set
# >>> test_images.shape
# (10000, 28, 28)
## number of labels in the test set
# >>> len(test_labels)
# 10000

## Use these functions later
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(
        "{} {:2.0f}% ({})".format(class_names[predicted_label],
        100*np.max(predictions_array),
        class_names[true_label]),
        color=color
    )

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

## Preprocessing
# save color range (pixel values) of the first image as a graph
# in this case, pixel values fall in the range of 0 to 255
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.savefig("output_images/t1_img_1st.png")

# scale pixel values to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# display the first 25 images from the training set and display the class name below each image
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.savefig("output_images/t1_training_set_class_names.png")

## Build the model
# setup the layers
model = keras.Sequential([
    # transform image from 2D (28 x 28 pixels square) array to 1D (28 * 28 = 784 pixels line) array
    keras.layers.Flatten(input_shape=(28, 28)),
    # first layer
    # 128 neurons
    # nn.relu (ReLU): Rectified Linear Unit (activation function)
    keras.layers.Dense(128, activation=tf.nn.relu),
    # second layer
    # 10 neurons
    # it returns 10 (same as the number of class) probability scores using softmax
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

## Compile the model
model.compile(
    # the way of updating model
    optimizer=tf.train.AdamOptimizer(),
    # this measures how accurate the model is during training (less is better)
    loss='sparse_categorical_crossentropy',
    # monitor the training and the testing steps
    # accuracy: the fraction of the correctly classified images
    metrics=['accuracy']
)

## Train the model
# epochs: how many times to feed the training data to the model
# it may cause overfitting if specifying too large number
model.fit(train_images, train_labels, epochs=5)

## Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

## Make predictions
predictions = model.predict(test_images)

# probabilities of the image that belongs to each class
# >>> predictions[0]
# array([3.1923636e-07, 2.1591529e-07, 3.0729248e-07, 5.9945108e-08,
#        5.7742932e-07, 1.0586625e-02, 1.8431626e-06, 5.0102264e-02,
#        2.1546361e-05, 9.3928629e-01], dtype=float32)
# >>> np.argmax(predictions[0])
# 9
# >>> test_labels[0]
# 9

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 10
num_cols = 6
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    # print(i)
    plt.subplot(num_rows, 2*num_cols, (2*i)+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, (2*i)+2)
    plot_value_array(i, predictions, test_labels)

plt.savefig("output_images/t1_result.png")
