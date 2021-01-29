import os
import six
import sys
import math
import ntpath
import random
import argparse
import threading
import numpy as np
import tensorflow as tf
from datetime import datetime


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if six.PY3 and isinstance(value, six.text_type):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer,
        label_type, label_rotation, label_perspective_x,
        label_perspective_y, height, width):
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label_type': _int64_feature(label_type),
        'image/class/label_rotation': _int64_feature(label_rotation),
        'image/class/label_perspective_x': _int64_feature(label_perspective_x),
        'image/class/label_perspective_y': _int64_feature(label_perspective_y),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


def _process_image(filename):
    # Read the image file.
    with tf.io.gfile.GFile(filename, 'rb') as f:
        image_data = f.read()

    # Decode the RGB JPEG.
    image = tf.io.decode_jpeg(image_data, channels=3)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    if height > width:
        width = int(299 * (width / height))
        height = 299
    else:
        height = int(299 * (height / width))
        width = 299

    image = tf.image.resize(image, [height, width],
        method=tf.image.ResizeMethod.LANCZOS3, antialias=True)

    image = tf.cast(tf.round(image), tf.uint8)

    image_data = tf.io.encode_jpeg(image).numpy()

    return image_data, height, width


def _process_image_files_batch(thread_index,
        ranges, output_directory, name, filenames, num_shards):
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name,
        # e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(output_directory, output_filename)
        writer = tf.io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1],
                                   dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            only_file = ntpath.basename(filename)
            label_type = ord(only_file[0]) - 64
            label_rotation = ord(only_file[1]) - 64
            label_perspective_x = ord(only_file[2]) - 64
            label_perspective_y = ord(only_file[3]) - 64

            image_buffer, height, width = _process_image(filename)

            example = _convert_to_example(only_file, image_buffer, label_type,
                                          label_rotation, label_perspective_x,
                                          label_perspective_y, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in'
                      ' thread batch.' % (datetime.now(), thread_index,
                      counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_shards))
    sys.stdout.flush()


def _process_image_files(num_threads,
        output_directory, name, filenames, num_shards):
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges,
                output_directory, name, filenames, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir):
    filenames = tf.io.gfile.glob('{}/*.jpg'.format(data_dir))

    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]

    print('Found {} .jpg files {}.'.format(len(filenames), data_dir))

    return filenames


def process_dataset(name, output_directory, directory):
    filenames = _find_image_files(directory)
    num_shards = math.ceil(len(filenames) / 400)
    _process_image_files(1, output_directory, name, filenames, num_shards)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--train_directory", default=r"C:\Users\abyerly"
        + r"\OneDrive - Brunel University London\micro_pcb_dataset\train_coded")
    p.add_argument("--test_directory", default=r"C:\Users\abyerly"
        + r"\OneDrive - Brunel University London\micro_pcb_dataset\test_coded")
    p.add_argument("--output_directory", default=r"C:\Users\abyerly"
        + r"\OneDrive - Brunel University London"
        + r"\micro_pcb_dataset\tfrecords\299")
    a = p.parse_args()

    if not os.path.isdir(a.output_directory):
        os.makedirs(a.output_directory)

    print('Saving processed files to {}'.format(a.output_directory))

    process_dataset('train', a.output_directory, a.train_directory)
    process_dataset('test', a.output_directory, a.test_directory)
