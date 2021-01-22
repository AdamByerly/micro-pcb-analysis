import tensorflow as tf
from input.micro_pcb_input_pipeline_base import MicroPCBBase


class MicroPCB(MicroPCBBase):
    def __init__(self, output, data_dir, image_size, batch_size,
            train_rotations_to_use=None, train_perspectives_x_to_use=None,
            train_perspectives_y_to_use=None, test_rotations_to_use=None,
            test_perspectives_x_to_use=None, test_perspectives_y_to_use=None,
            generate_missing_rotations=False,
            generate_missing_perspectives=False):
        MicroPCBBase.__init__(self, output, data_dir, image_size, batch_size)

        self._train_rotations_to_use = [1, 2, 3, 4, 5] \
            if train_rotations_to_use is None \
            else train_rotations_to_use
        self._train_perspectives_x_to_use = [1, 2, 3, 4, 5] \
            if train_perspectives_x_to_use is None \
            else train_perspectives_x_to_use
        self._train_perspectives_y_to_use = [1, 2, 3, 4, 5] \
            if train_perspectives_y_to_use is None \
            else train_perspectives_y_to_use

        self._missing_rotations = list({1, 2, 3, 4, 5}
            - set(self._train_rotations_to_use))
        self._missing_train_perspectives_x_to_use = list({1, 2, 3, 4, 5}
            - set(self._train_perspectives_x_to_use))
        self._missing_train_perspectives_y_to_use = list({1, 2, 3, 4, 5}
            - set(self._train_perspectives_y_to_use))

        self._test_rotations_to_use = [1, 2, 3, 4, 5] \
            if test_rotations_to_use is None \
            else test_rotations_to_use
        self._test_perspectives_x_to_use = [1, 2, 3, 4, 5] \
            if test_perspectives_x_to_use is None \
            else test_perspectives_x_to_use
        self._test_perspectives_y_to_use = [1, 2, 3, 4, 5] \
            if test_perspectives_y_to_use is None \
            else test_perspectives_y_to_use

        self._generate_missing_rotations = generate_missing_rotations
        self._generate_missing_perspectives = generate_missing_perspectives

    def get_training_image_count(self):
        return 4 * self.get_class_count() \
               * len(self._train_rotations_to_use) \
               * len(self._train_perspectives_x_to_use) \
               * len(self._train_perspectives_y_to_use)

    def get_validation_image_count(self):
        return self.get_class_count() \
               * len(self._test_rotations_to_use) \
               * len(self._test_perspectives_x_to_use) \
               * len(self._test_perspectives_y_to_use)

    # noinspection PyUnusedLocal
    def _train_filter(self, image, label_type, label_rotation,
               label_perspective_x, label_perspective_y, width, height):
        return tf.size(tf.sets.intersection([[label_rotation]],
                    [self._train_rotations_to_use])) > 0 \
            and tf.size(tf.sets.intersection([[label_perspective_x]],
                    [self._train_perspectives_x_to_use])) > 0 \
            and tf.size(tf.sets.intersection([[label_perspective_y]],
                    [self._train_perspectives_y_to_use])) > 0

    # noinspection PyUnusedLocal
    def _test_filter(self, image, label_type, label_rotation,
               label_perspective_x, label_perspective_y, width, height):
        return tf.size(tf.sets.intersection([[label_rotation]],
                    [self._test_rotations_to_use])) > 0 \
            and tf.size(tf.sets.intersection([[label_perspective_x]],
                    [self._test_perspectives_x_to_use])) > 0 \
            and tf.size(tf.sets.intersection([[label_perspective_y]],
                    [self._test_perspectives_y_to_use])) > 0

    def _train_augment(self, image, label, width, height, label_rotation,
               label_perspective_x, label_perspective_y):
        if self._generate_missing_rotations:
            image, label, width, height, label_rotation, \
                label_perspective_x, label_perspective_y = \
                self._rotate(image, label, width, height,
                label_rotation, label_perspective_x, label_perspective_y)
        if self._generate_missing_perspectives:
            image, label, width, height, label_rotation, \
                label_perspective_x, label_perspective_y = \
                self._perspective_warp(image, label, width, height,
                label_rotation, label_perspective_x, label_perspective_y)
        image, label, width, height, label_rotation, \
            label_perspective_x, label_perspective_y = \
            MicroPCBBase._translate(image, label, width, height,
            label_rotation, label_perspective_x, label_perspective_y)
        return image, label, width, height, label_rotation, \
               label_perspective_x, label_perspective_y

    # Assumes neutral is present, the missing rotations are symmetric,
    #  and if a shallow angle is missing, so is the wider angle.
    # i.e. if 1 is missing, so is 5; if 2 is missing so is 4
    #      if 2 is missing, so is 1; if 5 is missing so is 4
    def _rotate(self, image, label, width, height, label_rotation,
               label_perspective_x, label_perspective_y):
        if len(self._missing_rotations) < 1:
            return image, label, width, height, label_rotation, \
               label_perspective_x, label_perspective_y

        # Only missing the wide angles, so 40% chance to augment
        if len(self._missing_rotations) < 3:
            augment = tf.cond(tf.random.uniform(
                shape=[], maxval=5, dtype=tf.int32) > 2, lambda: True,
                lambda: False)
        # Missing shallow and wide angles, so 80% chance to augment
        else:
            augment = tf.cond(tf.random.uniform(
                shape=[], maxval=5, dtype=tf.int32) > 0, lambda: True,
                lambda: False)

        if not augment:
            return image, label, width, height, label_rotation, \
                   label_perspective_x, label_perspective_y

        def get_shallow_angle():
            return tf.cond(tf.random.uniform(
                shape=[], maxval=2, dtype=tf.int32) > 0,
                lambda: tf.random.normal([],
                            mean=-12.39499862, stddev=3.590359044),
                lambda: tf.random.normal([],
                            mean=14.73233285, stddev=4.462176094))

        def get_wide_angle():
            return tf.cond(tf.random.uniform(
                shape=[], maxval=2, dtype=tf.int32) > 0,
                lambda: tf.random.normal([],
                            mean=-21.31424765, stddev=4.504805275),
                lambda: tf.random.normal([],
                            mean=24.31108271, stddev=5.271388302))

        # This is a neutral rotation and we are missing all non-neutral
        if label_rotation == 3 and 2 in self._missing_rotations:
            angle = tf.cond(tf.random.uniform(
                shape=[], maxval=2, dtype=tf.int32) > 0,
                lambda: get_shallow_angle(),
                lambda: get_wide_angle())
        # This is a neutral rotation and we are missing only wide angles
        elif label_rotation == 3 and 1 in self._missing_rotations:
            angle = get_wide_angle()
        # This is a left shallow rotation (and we are missing only wide angles)
        elif label_rotation == 2:
            angle = tf.random.normal([],
                mean=-8.91924903, stddev=4.504805275)
        # This is a right shallow rotation (and we are missing only wide angles)
        else:
            angle = tf.random.normal([],
                mean=9.578749856494, stddev=5.271388302)

        new_image, width, height = tf.numpy_function(
            MicroPCBBase._image_rotate_random_py_func, (image, angle),
            (tf.float32, tf.int32, tf.int32))
        return new_image, label, width, height, label_rotation, \
               label_perspective_x, label_perspective_y

    # Assumes neutral is present, the missing perspectives are symmetric,
    #  and if a near perspective is missing, so is the far perspective.
    def _perspective_warp(self, image, label, width, height, label_rotation,
               label_perspective_x, label_perspective_y):
        if len(self._missing_train_perspectives_x_to_use) < 1:
            return image, label, width, height, label_rotation, \
               label_perspective_x, label_perspective_y

        # Only missing the far perspectives, so 40% chance to augment x
        if len(self._missing_train_perspectives_x_to_use) < 3:
            augment_x = tf.cond(tf.random.uniform(
                shape=[], maxval=5, dtype=tf.int32) > 2, lambda: True,
                lambda: False)
        # Missing near and far perspectives, so 80% chance to augment x
        else:
            augment_x = tf.cond(tf.random.uniform(
                shape=[], maxval=5, dtype=tf.int32) > 0, lambda: True,
                lambda: False)

        # Only missing the far perspectives, so 40% chance to augment y
        if len(self._missing_train_perspectives_y_to_use) < 3:
            augment_y = tf.cond(tf.random.uniform(
                shape=[], maxval=5, dtype=tf.int32) > 2, lambda: True,
                lambda: False)
        # Missing near and far perspectives, so 80% chance to augment y
        else:
            augment_y = tf.cond(tf.random.uniform(
                shape=[], maxval=5, dtype=tf.int32) > 0, lambda: True,
                lambda: False)

        if not augment_x and not augment_y:
            return image, label, width, height, label_rotation, \
                   label_perspective_x, label_perspective_y

        def get_perspective_change(amount, stddev):
            return tf.cond(tf.random.uniform(
                shape=[], maxval=2, dtype=tf.int32) > 0,
                lambda: tf.random.normal([], mean=-1*amount, stddev=stddev),
                lambda: tf.random.normal([], mean=1*amount, stddev=stddev))

        change_x = 0.
        if augment_x:
            # This is a neutral perspective and we are missing all non-neutral
            if label_perspective_x == 3 and \
                    2 in self._missing_train_perspectives_x_to_use:
                change_x = tf.cond(tf.random.uniform(
                    shape=[], maxval=2, dtype=tf.int32) > 0,
                    lambda: get_perspective_change(6.8513193, 5.3184557),
                    lambda: get_perspective_change(2.7788302, 4.323846))
            # This is a neutral perspective and we are missing only far
            elif label_perspective_x == 3 and \
                    1 in self._missing_train_perspectives_x_to_use:
                change_x = get_perspective_change(6.8513193, 5.3184557)
            # This is a left near (and we are missing only far)
            elif label_perspective_x == 2:
                change_x = tf.random.normal([],
                            mean=-2.7788302, stddev=4.323846)
            # This is a right near (and we are missing only far)
            else:
                change_x = tf.random.normal([],
                            mean=2.7788302, stddev=4.323846)

        change_y = 0.
        if augment_y:
            # This is a neutral perspective and we are missing all non-neutral
            if label_perspective_y == 3 and \
                    2 in self._missing_train_perspectives_y_to_use:
                change_y = tf.cond(tf.random.uniform(
                    shape=[], maxval=2, dtype=tf.int32) > 0,
                    lambda: get_perspective_change(7.6808512, 5.3184557),
                    lambda: get_perspective_change(3.1152805, 4.323846))
            # This is a neutral perspective and we are missing only far
            elif label_perspective_y == 3 and \
                    1 in self._missing_train_perspectives_y_to_use:
                change_y = get_perspective_change(7.6808512, 5.3184557)
            # This is a left near (and we are missing only far)
            elif label_perspective_y == 2:
                change_y = tf.random.normal([],
                            mean=-3.1152805, stddev=4.323846)
            # This is a right near (and we are missing only far)
            else:
                change_y = tf.random.normal([],
                            mean=3.1152805, stddev=4.323846)

        new_image, width, height = tf.numpy_function(
            MicroPCBBase._image_perspective_warp_random_py_func,
            (image, change_x, change_y), (tf.float32, tf.int32, tf.int32))
        return new_image, label, width, height, label_rotation, \
               label_perspective_x, label_perspective_y
