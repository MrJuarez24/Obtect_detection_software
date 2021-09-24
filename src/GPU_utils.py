import tensorflow as tf
import os
#import keras.backend as k

def tensorflow_2_x_dark_magic_to_restrict_memory_use(GPU_TO_USE):

    #tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)

    if gpus:
        # Restrict TensorFlow to only use GPU GPU_TO_USE.
        try:
            tf.config.experimental.set_visible_devices(gpus[GPU_TO_USE], 'GPU')             # Set as visible only GPU GPU_TO_USE
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)                # Set memory grow.
            #tf.config.experimental.set_virtual_device_configuration(
            #    gpus[GPU_TO_USE],
            #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])    # Set maximum memory size on GPU_TO_USE to 10GB
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
