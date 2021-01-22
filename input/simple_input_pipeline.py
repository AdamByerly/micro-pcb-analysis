import os
import abc
import tensorflow as tf
from input.input_pipeline_base import InputPipelineBase
# noinspection PyProtectedMember
from tensorflow.python.ops.image_ops_impl import _ImageDimensions

PARALLEL_INPUT_CALLS = 16
TRAINING_BUFFER_SIZE = 1024


class SimpleInputPipeline(InputPipelineBase):
    def __init__(self, output, data_dir, classes, image_size, batch_size):
        InputPipelineBase.__init__(self, output, batch_size)
        self._data_dir   = data_dir
        self._classes    = classes
        self._image_size = image_size

        self._train_steps = \
            self.get_training_image_count() // self._batch_size
        self._validation_steps = \
            self.get_validation_image_count() // self._batch_size

        self._initialized = True

    def get_class_count(self):
        return self._classes

    @abc.abstractmethod
    def get_training_image_count(self):
        pass

    @abc.abstractmethod
    def get_validation_image_count(self):
        pass

    def get_training_dataset(self):
        with tf.name_scope("train_input"):
            dataset = self._get_pipeline_start("train-*")
            dataset = dataset.shuffle(buffer_size=TRAINING_BUFFER_SIZE)
            dataset = dataset.map(self._decode_jpeg,
                num_parallel_calls=PARALLEL_INPUT_CALLS)
            dataset = dataset.map(self._train_augment,
                num_parallel_calls=PARALLEL_INPUT_CALLS)
            return self._get_pipeline_end(dataset)

    def get_validation_dataset(self):
        with tf.name_scope("validation_input"):
            dataset = self._get_pipeline_start("test-*")
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

    @staticmethod
    def _train_augment(image, label, width, height):
        begin = [0, 0, 0]
        size  = [-1, -1, 3]

        # get values between 0.0 and 0.05 as how much to adjust the image by
        rand_vals     = tf.random.uniform([2])*0.05
        width_adjust  = tf.cast(tf.cast(width,
                            tf.float32)*rand_vals[0], tf.int32)
        height_adjust = tf.cast(tf.cast(height,
                            tf.float32)*rand_vals[1], tf.int32)

        rand_dirs = tf.random.uniform([2], minval=0, maxval=2, dtype=tf.int32)
        if rand_dirs[0] > 0 :
            begin[0] = height_adjust
        else:
            size[0] = height - height_adjust
        if rand_dirs[1] > 0:
            begin[1] = width_adjust
        else:
            size[1] = width - width_adjust

        image = tf.slice(image, begin, size)
        return image, label, width, height

    # noinspection PyUnusedLocal
    def _prepare_image(self, image, label, width, height):
        # This resizing operation may distort the images because the aspect
        # ratio is not respected.
        # image = tf.image.resize(image,
        #     [self._image_size, self._image_size],
        #     method=tf.image.ResizeMethod.LANCZOS3, antialias=True)
        image = tf.image.resize_with_pad(
            image, self._image_size, self._image_size,
            method=tf.image.ResizeMethod.LANCZOS3, antialias=True)

        image.set_shape([self._image_size, self._image_size, 3])

        return tf.squeeze(image), \
               tf.squeeze(tf.one_hot(label, self.get_class_count()))

    # noinspection PyUnusedLocal
    @staticmethod
    def _decode_jpeg(image, label, bbox):
        image = tf.image.decode_jpeg(image, channels=3)

        image_height, image_width, _ = _ImageDimensions(image, rank=3)

        if tf.size(bbox) < 1:
            height = image_height
            width  = image_width
        else:
            bbox = bbox[0][0]
            height = tf.cast((tf.squeeze(bbox[2]) - tf.squeeze(bbox[0]))
                        * tf.cast(image_height, tf.float32), tf.int32)
            width  = tf.cast((tf.squeeze(bbox[3]) - tf.squeeze(bbox[1]))
                        * tf.cast(image_width, tf.float32), tf.int32)
            y      = tf.cast(tf.squeeze(bbox[0])
                        * tf.cast(image_height, tf.float32), tf.int32)
            x       = tf.cast(tf.squeeze(bbox[1])
                        * tf.cast(image_width, tf.float32), tf.int32)
            image = tf.image.crop_to_bounding_box(image, y, x, height, width)

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        return image, label, width, height

    @staticmethod
    def _image_center(image, label):
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image, label

    @staticmethod
    def _parse_example_proto(example_serialized):
        feature_map = {
            'image/encoded': tf.io.FixedLenFeature([],
                dtype=tf.string, default_value=''),
            'image/class/label': tf.io.FixedLenFeature([1],
                dtype=tf.int64, default_value=-1),
            'image/class/text': tf.io.FixedLenFeature([],
                dtype=tf.string, default_value=''),
            'image/class/synset': tf.io.FixedLenFeature([],
                dtype=tf.string, default_value=''),
            'image/filename': tf.io.FixedLenFeature([],
                dtype=tf.string, default_value='')}
        sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
        feature_map.update({k: sparse_float32 for k in [
            'image/object/bbox/xmin', 'image/object/bbox/ymin',
            'image/object/bbox/xmax', 'image/object/bbox/ymax']})
        features = tf.io.parse_single_example(example_serialized, feature_map)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)
        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
        bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])
        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(bbox, [0, 2, 1])
        return features['image/encoded'], label, bbox
