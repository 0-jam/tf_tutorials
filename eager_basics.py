import tensorflow as tf
tf.enable_eager_execution()

## Calculating tensors
# tf.Tensor(3, shape=(), dtype=int32)
print("tf.add(1, 2) =", tf.add(1, 2))
# tf.Tensor([4 6], shape=(2,), dtype=int32)
print("tf.add([1, 2], [3, 4]) =", tf.add([1, 2], [3, 4]))
# tf.Tensor(25, shape=(), dtype=int32)
print("tf.square(5) =", tf.square(5))
# tf.Tensor(6, shape=(), dtype=int32)
print("tf.reduce_sum([1, 2, 3]) =", tf.reduce_sum([1, 2, 3]))
# tf.Tensor(b'aGVsbG8gd29ybGQ', shape=(), dtype=string)
print("tf.encode_base64('hello world'):", tf.encode_base64('hello world'))

# Operator overloading
# tf.Tensor(13, shape=(), dtype=int32)
print("tf.square(2) + tf.square(3) =", tf.square(2) + tf.square(3))

# Show a shape and a datatype of tensor
print("x = tf.matmul([[1]], [[2, 3]])")
x = tf.matmul([[1]], [[2, 3]])
print("Shape of x:", x.shape)
print("Data type of x:", x.dtype)

## NumPy conpatibility
import numpy as np

ndarray = np.ones([3, 3])
# array([[1., 1., 1.],
#        [1., 1., 1.],
#        [1., 1., 1.]])
print("np.ones([3, 3]):", ndarray)

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
# <tf.Tensor: id=3, shape=(3, 3), dtype=float64, numpy=
# array([[42., 42., 42.],
#        [42., 42., 42.],
#        [42., 42., 42.]])>
print("42 times multiplied ndarray by TensorFlow:", tensor)

print("And NumPy operations convert Tensors to numpy arrays automatically")
# array([[43., 43., 43.],
#        [43., 43., 43.],
#        [43., 43., 43.]])
print("1 is added to tensor by NumPy:", np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
# array([[42., 42., 42.],
#        [42., 42., 42.],
#        [42., 42., 42.]])
print(tensor.numpy())

## GPU acceleration
# [3, 3] array with random values
x = tf.random_uniform([3, 3])

# Show which device is used for TensorFlow acceleration
# TensorFlow automatically decides the GPU or CPU for operation
print("Is there a GPU available:", tf.test.is_gpu_available())
# '/job:localhost/replica:0/task:0/device:CPU:0'
print("Tensor device is:", x.device)
# print("Is the Tensor on GPU #0", x.device.endswith("GPU:0"))

## Explicit Device Placement
import time
# Execute on CPU
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    start_time = time.time()
    tf.matmul(x, x)
    print("Elapsed time (CPU): {} sec".format(time.time() - start_time))

# Execute on GPU if available
if tf.test.is_gpu_available():
    print("On GPU:")
    with tf.device("GPU:0"):
        x = tf.random_uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        start_time = time.time()
        tf.matmul(x, x)
        print("Elapsed time (GPU): {} sec".format(time.time() - start_time))

## Create a source dataset
# [1, 2, 3, 4, 5, 6]
ds_tensors = tf.data.Dataset.from_tensor_slices(list(range(1, 7)))

# Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()

with open("output_images/eager.csv", "w") as file:
    file.write("\n".join(["Line 1", "Line 2", "Line 3"]))

ds_file = tf.data.TextLineDataset(filename)

# Apply transformations
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

print("Elements of ds_tensors:")
# tf.Tensor([4 1], shape=(2,), dtype=int32)
# tf.Tensor([16 25], shape=(2,), dtype=int32)
# tf.Tensor([36  9], shape=(2,), dtype=int32)
for x in ds_tensors:
    print(x)

print("Elements of ds_file:")
# BUG?: It printed nothing in my environment ...
for x in ds_file:
    print(x)
