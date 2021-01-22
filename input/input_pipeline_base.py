# Copyright 2020 Adam Byerly. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import abc
from constructs.loggable import Loggable


class InputPipelineBase(Loggable, metaclass=abc.ABCMeta):
    def __init__(self, output, batch_size):
        """
        Before leaving the __init__ method, subclasses should call
        self._initialize(), which will in turn call self._load_training_data()
        and self._load_validation_data().
        """
        Loggable.__init__(self)
        self._output            = output
        self._batch_size        = batch_size
        self._initialized       = False
        self._train_steps       = None
        self._validation_steps  = None

    def get_train_steps(self):
        self._initialize()
        return self._train_steps

    def get_validation_steps(self):
        self._initialize()
        return self._validation_steps

    @abc.abstractmethod
    def get_training_dataset(self):
        """
        This method should return a tf.data.Dataset for the training data.
        This dataset should be ready for consumption and thus be batched and
        have all processing applied.
        """
        pass

    @abc.abstractmethod
    def get_validation_dataset(self):
        """
        This method should return a tf.data.Dataset for the validation data.
        This dataset should be ready for consumption and thus be batched and
        have all processing applied.
        """
        pass

    @abc.abstractmethod
    def _load_training_data(self):
        """
        This method should return a tf.data.Dataset for the training data.
        This dataset must not be batched and should not have any processing
        applied.
        """
        pass

    @abc.abstractmethod
    def _load_validation_data(self):
        """
        This method should return a tf.data.Dataset for the validation data.
        This dataset must not be batched and should not have any processing
        applied.
        """
        pass

    def _initialize(self):
        if self._initialized:
            return

        self._initialized = True

        self._output.print_msg("Determining the number of training steps...")

        dataset = self._load_training_data()
        dataset = dataset.batch(self._batch_size)

        it = iter(dataset)

        try:
            self._train_steps = 0
            while True:
                next(it)
                self._train_steps += 1
        except StopIteration:
            pass

        self._output.print_msg("Determining the number of validation steps...")

        dataset = self._load_validation_data()
        dataset = dataset.batch(self._batch_size)

        it = iter(dataset)

        try:
            self._validation_steps = 0
            while True:
                next(it)
                self._validation_steps += 1
        except StopIteration:
            pass
