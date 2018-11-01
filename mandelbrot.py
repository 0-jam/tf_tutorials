### Visualizing the Mandelbrot set
## Import libraries for simulation
import tensorflow as tf
import numpy as np

## Imports for visualization
import PIL.Image
# from io import BytesIO
from IPython.display import Image, display

## Display an array of iteration counts as a colorful picture of a fractal
def drawFractal(arr):
    a_cyclic = (np.pi * 2 * arr / 20.0).reshape(list(arr.shape) + [1])
    img = np.concatenate(
        # set color
        [
            10 + 20 * np.cos(a_cyclic),
            30 + 50 * np.sin(a_cyclic),
            155 - 80 * np.cos(a_cyclic),
        ],
        2
    )

    img[arr == arr.max()] = 0
    arr = img
    arr = np.uint8(np.clip(arr, 0, 255))

    # f = BytesIO()
    # display(Image(data = f.getvalue()))
    return PIL.Image.fromarray(arr)

session = tf.InteractiveSession()

# Use NumPy to create a 2D array of complex numbers
# mgrid: meshgrid
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Z = X + 1j * Y

xs = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, tf.float32))

tf.global_variables_initializer().run()

# Compute the new values of z: z^2 + x
zs_ = zs * zs + xs

# Have we diverged with this new value?
not_diverged = tf.abs(zs_) < 4

# Operation to update the zs and the iteration count.
# Note:
#   We keep computing zs after they diverge! This is very wasteful!
#   There are better, if a little less simple, ways to do this.
step = tf.group(
    zs.assign(zs_),
    ns.assign_add(tf.cast(not_diverged, tf.float32))
)

for i in range(200):
    step.run()

image = drawFractal(ns.eval())
image.save('output_images/mandelbrot.png')
