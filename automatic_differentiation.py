import tensorflow as tf
tf.enable_eager_execution()
import numpy

## Show TensorFlow version
print("Using TensorFlow", tf.__version__)

# Shorthand for some symbols
tfe = tf.contrib.eager

## Derivatives of a function
import math
pi = math.pi

def sin_square(radius):
    return tf.square(tf.sin(radius))

# Throw exception if sin_square(pi / 2) is not 1.0
# sin(pi / 2) = 1.0
assert sin_square(pi / 2).numpy() == 1.0

# grad_square will return a list of derivatives of f
# with respect to its arguments. Since f() has a single argument,
# grad_square will return a list with a single element.
grad_square = tfe.gradients_function(sin_square)
# 8.742278e-08
assert tf.abs(grad_square(pi / 2)[0]).numpy() < 1e-7

## Higher-order gradients
def grad(function):
    return lambda x: tfe.gradients_function(function)(x)[0]

# 100 points between -2π (-360 degree) and +2π (+360 degree)
radius = tf.lin_space(-2 * pi, 2 * pi, 100)

import matplotlib.pyplot as plt

plt.plot(radius, sin_square(radius), label="sin_square")
plt.plot(radius, grad(sin_square)(radius), label="first derivative")
plt.plot(radius, grad(grad(sin_square))(radius), label="second derivative")
plt.plot(radius, grad(grad(grad(sin_square)))(radius), label="third derivative")
plt.legend()
plt.show()

## Gradient tapes
# f(x) = x^y
def f(x, y):
    out = 1
    # Must use range(int(y)) instead of range(y) in Python 3 when using TensorFlow <= 1.10.x
    for i in range(int(y)):
        out = tf.multiply(out, x)

    return out

# f'(x) = dx/dy(x^y) = y * x
def grad_f(x, y):
    return tfe.gradients_function(f)(x, y)[0]

x, y = 3.0, 2
# f(x) = 3^2
fx = f(x, y).numpy()
assert fx == 9.0
print("f(x) = {}^{} = {}".format(x, y, fx))
# f'(x) = dx/dy(3^2) = 2 * 3
grad_fx = grad_f(x, y).numpy()
assert grad_fx == 6.0
print("f'(x) = dx/dy({}^{}) = {}".format(x, y, grad_fx))

x, y = 4.0, 3
# f(x) = 4^3
fx = f(x, y).numpy()
assert fx == 64.0
print("f(x) = {}^{} = {}".format(x, y, fx))
# f'(x) = dx/dy(4^3) = 3 * 4^2
grad_fx = grad_f(x, y).numpy()
assert grad_fx == 48.0
print("f'(x) = dx/dy({}^{}) = {}".format(x, y, grad_fx))

x = tf.ones((2, 2))

# Record all intermediate values computed in a function
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    # The sum of elements across dimensions of a tensor
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# Use the same tape to compute the derivative of z with respect to the intermediate value y
dz_dy = tape.gradient(z, y)
assert dz_dy.numpy() == 8.0
print("x =", x.numpy())
print("y = tf.reduce_sum(x) =", y)
print("z = tf.multiply(y, y) = y^2 =", z)
print("dz/dy = 2 * y =", dz_dy.numpy())

# Derivative of z with respect to the original input tensor x
dz_dx = tape.gradient(z, x)
for i in [0, 1]:
    for j in [0, 1]:
        assert dz_dx[i][j].numpy() == 8.0
        print("dz/dy[{}][{}] = {}".format(i, j, dz_dy.numpy()))

## Higher-order gradients
# Convert the Python 1.0 to a Tensor object
x = tf.constant(1.0)

with tf.GradientTape() as tape1:
    with tf.GradientTape() as tape2:
        tape2.watch(x)
        y = x * x * x

    # Compute the gradient inside the 't' context manager
    # which means computation is differentiable as well
    dy_dx = tape2.gradient(y, x)
d2y_dx2 = tape1.gradient(dy_dx, x)

# y = x^3
# dy/dx = 3 * x^2
# dy/dx(dy/dx) = 3 * 2 * x
print("f(x) = x^3")
print("y = f(1) = 1^3 = 1")
assert dy_dx.numpy() == 3.0
print("dy/dx = 3 * 1^2 =", dy_dx.numpy())
assert d2y_dx2.numpy() == 6.0
print("dy/dx(dy/dx) = 3 * 2 * 1 =", d2y_dx2.numpy())
