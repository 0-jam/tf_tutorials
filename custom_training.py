import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager

import numpy

## Show TensorFlow version
print("Using TensorFlow", tf.__version__)

## 10 dimensions tensor
x = tf.zeros([10, 10])
# Add 2 to all elements in tensor x
x += 2
print(x)

# In the tutorial, v = tf.Variable(1.0)
# but it returns follwing error:
# RuntimeError: tf.Variable not supported when eager execution is enabled. Please use tf.contrib.eager.Variable instead
v = tfe.Variable(1.0)
assert v.numpy() == 1.0
print("Value v =", v.numpy())

# Re-assign the value
v.assign(3.0)
assert v.numpy() == 3.0
print("Reassigned v =", v.numpy())

# Use v in a TensorFlow operation like tf.square() and reassign
v.assign(tf.square(v))
assert v.numpy() == 9.0
print("Squared v =", v.numpy())

## Example: Fitting a linear model
# Define the model
class Model(object):
    def __init__(self):
        # Initialize variable to (5.0, 0.0)
        # In practice, these should be initialized to random numbers
        # Weight?
        self.W = tfe.Variable(5.0)
        # bias?
        self.b = tfe.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b

model = Model()
# W * x + b = 5 * 3 + 0
assert model(3.0).numpy() == 15.0
print("W * x + b = 5 * 3 + 0 =", model(3.0).numpy())

# Define a loss function
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

# Synthesize the training data with some noise
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random_normal(shape = [NUM_EXAMPLES])
noise = tf.random_normal(shape = [NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

# Visualize where the model stands right now
import matplotlib.pyplot as plt
# Argument "c" means color
# 'b' is blue, 'r is red
plt.scatter(inputs, outputs, c = 'b')
plt.scatter(inputs, model(inputs), c = 'r')
plt.show()

print("Current loss:", loss(model(inputs), outputs).numpy())

# Define a training loop
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss(model(inputs), outputs)

    dW, db = tape.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

# Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)

    train(model, inputs, outputs, learning_rate=0.1)
    print("Epoch {:2d}: W = {:1.2f} b = {:1.2f}, loss = {:2.5f}".format(epoch, Ws[-1], bs[-1], current_loss))

# Plot it all
plt.clf()
plt.plot(epochs, Ws, 'r', epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--', [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true b'])
plt.show()
