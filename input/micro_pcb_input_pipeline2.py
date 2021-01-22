import tensorflow as tf
from input.micro_pcb_input_pipeline_base import MicroPCBBase

PARALLEL_INPUT_CALLS = 16
TRAINING_BUFFER_SIZE = 1024


class MicroPCBUseAllAndAugmentAll(MicroPCBBase):
    def __init__(self, output, data_dir, image_size, batch_size):
        MicroPCBBase.__init__(self, output, data_dir, image_size, batch_size)

    def get_training_image_count(self):
        return 500 * self.get_class_count()

    def get_validation_image_count(self):
        return 125 * self.get_class_count()

    def _train_augment(self, image, label, width, height, label_rotation,
               label_perspective_x, label_perspective_y):
        image, label, width, height, label_rotation, \
            label_perspective_x, label_perspective_y = \
            MicroPCBUseAllAndAugmentAll._rotate(image, label, width, height,
            label_rotation, label_perspective_x, label_perspective_y)
        image, label, width, height, label_rotation, \
            label_perspective_x, label_perspective_y = \
            MicroPCBUseAllAndAugmentAll._perspective_warp(image, label, width, height,
            label_rotation, label_perspective_x, label_perspective_y)
        image, label, width, height, label_rotation, \
            label_perspective_x, label_perspective_y = \
            MicroPCBBase._translate(image, label, width, height,
            label_rotation, label_perspective_x, label_perspective_y)
        return image, label, width, height, label_rotation, \
               label_perspective_x, label_perspective_y

    # noinspection PyUnusedLocal
    @staticmethod
    def _rotate(image, label, width, height, label_rotation,
               label_perspective_x, label_perspective_y):

        if label_rotation == 1:
            # This is a left wide rotation
            angle = tf.random.normal([], mean=0, stddev=4.504805275)
        elif label_rotation == 2:
            # This is a left shallow rotation
            angle = tf.random.normal([], mean=0, stddev=3.590359044)
        elif label_rotation == 3:
            # This is a neutral rotation
            angle = tf.random.normal([], mean=0, stddev=2.04823414)
        elif label_rotation == 4:
            # This is a right shallow rotation
            angle = tf.random.normal([], mean=0, stddev=4.462176094)
        else:
            # This is a right wide rotation
            angle = tf.random.normal([], mean=0, stddev=5.271388302)

        new_image, width, height = tf.numpy_function(
            MicroPCBBase._image_rotate_random_py_func, (image, angle),
            (tf.float32, tf.int32, tf.int32))
        return new_image, label, width, height, label_rotation, \
               label_perspective_x, label_perspective_y

    # noinspection PyUnusedLocal
    @staticmethod
    def _perspective_warp(image, label, width, height, label_rotation,
               label_perspective_x, label_perspective_y):

        if label_perspective_x == 1:
            # This is an above, far perspective
            change_x = tf.random.normal([], mean=0, stddev=6.2713035)
        elif label_perspective_x == 2:
            # This is an above, near perspective
            change_x = tf.random.normal([], mean=0, stddev=5.0528257)
        elif label_perspective_x == 3:
            # This is a neutral perspective
            change_x = tf.random.normal([], mean=0, stddev=3.5090973)
        elif label_perspective_x == 4:
            # This is a below, near perspective
            change_x = tf.random.normal([], mean=0, stddev=3.5948663)
        else:
            # This is a below, far perspective
            change_x = tf.random.normal([], mean=0, stddev=4.3656079)

        if label_perspective_y == 1:
            # This is a left, far perspective
            change_y = tf.random.normal([], mean=0, stddev=5.595883)
        elif label_perspective_y == 2:
            # This is a left, near perspective
            change_y = tf.random.normal([], mean=0, stddev=4.5086355)
        elif label_perspective_y == 3:
            # This is a neutral perspective
            change_y = tf.random.normal([], mean=0, stddev=3.1311669)
        elif label_perspective_y == 4:
            # This is a right, near perspective
            change_y = tf.random.normal([], mean=0, stddev=3.2076986)
        else:
            # This is a right, far perspective
            change_y = tf.random.normal([], mean=0, stddev=3.8954312)

        new_image, width, height = tf.numpy_function(
            MicroPCBBase._image_perspective_warp_random_py_func,
            (image, change_x, change_y), (tf.float32, tf.int32, tf.int32))
        return new_image, label, width, height, label_rotation, \
               label_perspective_x, label_perspective_y
