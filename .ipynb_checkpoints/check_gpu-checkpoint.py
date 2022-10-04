from tensorflow.python.client import device_lib
import tensorflow as tf


def get():
    local_devices = device_lib.list_local_devices()
    for x in local_devices:
        if x.device_type == "GPU":
            print(x.name)


get()

print("You are using TensorFlow version", tf.__version__)
if len(tf.config.list_physical_devices("GPU")) > 0:
    print("You have a GPU enabled.")
else:
    print("Enable a GPU before running this notebook.")
