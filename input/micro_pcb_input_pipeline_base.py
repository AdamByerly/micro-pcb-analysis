import os
import abc
import cv2
import numpy as np
import tensorflow as tf
from input.input_pipeline_base import InputPipelineBase

PARALLEL_INPUT_CALLS = 16
TRAINING_BUFFER_SIZE = 1024


class MicroPCBBase(InputPipelineBase, metaclass=abc.ABCMeta):
    def __init__(self, output, data_dir, image_size, batch_size):
        InputPipelineBase.__init__(self, output, batch_size)
        self._data_dir   = data_dir
        self._image_size = image_size

    def _initialize(self):
        if self._initialized:
            return

        self._initialized = True

        self._train_steps = \
            self.get_training_image_count() // self._batch_size
        self._validation_steps = \
            self.get_validation_image_count() // self._batch_size

    @staticmethod
    def get_class_count():
        return 13

    @abc.abstractmethod
    def get_training_image_count(self):
        pass

    @abc.abstractmethod
    def get_validation_image_count(self):
        pass

    def get_training_dataset(self):
        with tf.name_scope("train_input"):
            dataset = self._get_pipeline_start("train-*")
            dataset = dataset.filter(self._train_filter)
            dataset = dataset.shuffle(buffer_size=TRAINING_BUFFER_SIZE)
            dataset = dataset.map(self._decode_jpeg,
                num_parallel_calls=PARALLEL_INPUT_CALLS)
            dataset = dataset.map(self._train_augment,
                num_parallel_calls=PARALLEL_INPUT_CALLS)
            return self._get_pipeline_end(dataset)

    def get_validation_dataset(self):
        with tf.name_scope("validation_input"):
            dataset = self._get_pipeline_start("test-*")
            dataset = dataset.filter(self._test_filter)
            dataset = dataset.map(self._decode_jpeg,
                num_parallel_calls=PARALLEL_INPUT_CALLS)
            return self._get_pipeline_end(dataset)

    def _load_training_data(self):
        return self._get_unparsed_data("train-*")

    def _load_validation_data(self):
        return self._get_unparsed_data("test-*")

    def _get_unparsed_data(self, file_pattern):
        tf_record_pattern = os.path.join(self._data_dir, file_pattern)
        dataset = tf.data.Dataset.list_files(
            file_pattern=tf_record_pattern, shuffle=True)
        dataset = tf.data.TFRecordDataset(dataset)
        return dataset

    def _get_pipeline_start(self, file_pattern):
        dataset = self._get_unparsed_data(file_pattern)
        dataset = dataset.map(self._parse_example_proto,
            num_parallel_calls=PARALLEL_INPUT_CALLS)
        return dataset

    def _get_pipeline_end(self, dataset):
        dataset = dataset.map(self._prepare_image,
            num_parallel_calls=PARALLEL_INPUT_CALLS)
        dataset = dataset.map(self._image_center,
            num_parallel_calls=PARALLEL_INPUT_CALLS)
        return dataset.prefetch(-1)

    def _train_filter(self, image, label_type, label_rotation,
               label_perspective_x, label_perspective_y, width, height):
        return True

    def _test_filter(self, image, label_type, label_rotation,
               label_perspective_x, label_perspective_y, width, height):
        return True

    @abc.abstractmethod
    def _train_augment(self, image, label, width, height, label_rotation,
               label_perspective_x, label_perspective_y):
        pass

    @staticmethod
    def _translate(image, label, width, height, label_rotation,
               label_perspective_x, label_perspective_y):
        begin = [0, 0, 0]
        size  = [-1, -1, 3]

        # get values between 0.0 and 0.05 as how much to adjust the image by
        rand_vals     = tf.random.uniform([2])*0.05
        width_adjust  = tf.cast(tf.cast(width,
                            tf.float32)*rand_vals[0], tf.int32)
        height_adjust = tf.cast(tf.cast(height,
                            tf.float32)*rand_vals[1], tf.int32)

        rand_dirs = tf.random.uniform([2], minval=0, maxval=2, dtype=tf.int32)
        if rand_dirs[0] > 0:
            begin[0] = height_adjust
        else:
            size[0] = height - height_adjust
        if rand_dirs[1] > 0:
            begin[1] = width_adjust
        else:
            size[1] = width - width_adjust

        image = tf.slice(image, begin, size)
        return image, label, width, height, label_rotation, \
               label_perspective_x, label_perspective_y

    @staticmethod
    def _image_rotate_random_py_func(image, angle):
        h, w, c = image.shape
        rot_mat = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)

        cos = abs(rot_mat[0][0])
        sin = abs(rot_mat[0][1])

        x_pad = int(((h*sin + w*cos) - w) / 2)
        y_pad = int(((h*cos + w*sin) - h) / 2)

        grown = cv2.copyMakeBorder(image, y_pad, y_pad+1, x_pad, x_pad+1,
            cv2.BORDER_REPLICATE)

        h, w, c = grown.shape

        rotated = cv2.warpAffine(grown, rot_mat, (w, h))

        w = tf.dtypes.cast(w, tf.int32)
        h = tf.dtypes.cast(h, tf.int32)
        return rotated, w, h

    @staticmethod
    def _image_perspective_warp_random_py_func(image, change_x, change_y):
        h, w, c = image.shape

        change_x_top, change_x_bot = (0, 0)
        if change_x < 0:
            change_x_top = -change_x / 2
        else:
            change_x_bot = change_x / 2

        change_y_left, change_y_right = (0, 0)
        if change_y < 0:
            change_y_left = -change_y / 2
        else:
            change_y_right = change_y / 2

        x1 = change_x_top * (w*.01)
        x2 = w - (change_x_top * (w*.01))
        x3 = change_x_bot * (w*.01)
        x4 = w - (change_x_bot * (w*.01))

        y1 = change_y_left * (h*.01)
        y2 = change_y_right * (h*.01)
        y3 = h - (change_y_left * (h*.01))
        y4 = h - (change_y_right * (h*.01))

        original = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        warped = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

        warp_matrix = cv2.getPerspectiveTransform(original, warped)
        warped = cv2.warpPerspective(image, warp_matrix, (w, h))

        w = tf.dtypes.cast(w, tf.int32)
        h = tf.dtypes.cast(h, tf.int32)
        return warped, w, h

    # noinspection PyUnusedLocal
    def _prepare_image(self, image, label, width, height, label_rotation,
               label_perspective_x, label_perspective_y):
        image = tf.image.resize_with_pad(
            image, self._image_size, self._image_size,
            method=tf.image.ResizeMethod.LANCZOS3, antialias=True)

        image.set_shape([self._image_size, self._image_size, 3])

        return tf.squeeze(image), \
               tf.squeeze(tf.one_hot(label, self.get_class_count()))

    # noinspection PyUnusedLocal
    @staticmethod
    def _decode_jpeg(image, label_type, label_rotation,
               label_perspective_x, label_perspective_y, width, height):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image, label_type, width, height, label_rotation, \
               label_perspective_x, label_perspective_y

    @staticmethod
    def _image_center(image, label):
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image, label

    @staticmethod
    def _parse_example_proto(example_serialized):
        feature_map = {
            'image/encoded'          : tf.io.FixedLenFeature([],
                                        dtype=tf.string, default_value=''),
            'image/width'            : tf.io.FixedLenFeature([],
                                        dtype=tf.int64, default_value=-1),
            'image/height'           : tf.io.FixedLenFeature([],
                                        dtype=tf.int64, default_value=-1),
            'image/class/label_type' : tf.io.FixedLenFeature([1],
                                        dtype=tf.int64, default_value=-1),
            'image/class/label_rotation': tf.io.FixedLenFeature([],
                                        dtype=tf.int64, default_value=-1),
            'image/class/label_perspective_x': tf.io.FixedLenFeature([],
                                        dtype=tf.int64, default_value=-1),
            'image/class/label_perspective_y': tf.io.FixedLenFeature([],
                                        dtype=tf.int64, default_value=-1),
            'image/filename'         : tf.io.FixedLenFeature([],
                                        dtype=tf.string, default_value='')}
        features            = tf.io.parse_single_example(
                                example_serialized, feature_map)
        label_type          = tf.cast(features['image/class/label_type'],
                                dtype=tf.int32)
        label_rotation      = tf.cast(features['image/class/label_rotation'],
                                dtype=tf.int32)
        label_perspective_x = tf.cast(features[
                                'image/class/label_perspective_x'],
                                dtype=tf.int32)
        label_perspective_y = tf.cast(features[
                                'image/class/label_perspective_y'],
                                dtype=tf.int32)
        width               = tf.cast(features['image/width'], dtype=tf.int32)
        height              = tf.cast(features['image/height'], dtype=tf.int32)

        return features['image/encoded'], label_type, label_rotation, \
               label_perspective_x, label_perspective_y, width, height
