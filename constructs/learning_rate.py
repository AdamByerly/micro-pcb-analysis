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

from constructs.loggable import Loggable
import tensorflow as tf


class ExponentialDecay(
        tf.keras.optimizers.schedules.ExponentialDecay, Loggable):
    pass


class CappedExponentialDecay(
        tf.keras.optimizers.schedules.ExponentialDecay, Loggable):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate,
            staircase=False, minimum_lr=0.0, name=None):
        tf.keras.optimizers.schedules.ExponentialDecay.__init__(self,
            initial_learning_rate, decay_steps, decay_rate, staircase, name)
        Loggable.__init__(self)
        self.minimum_lr = minimum_lr

    def __call__(self, step):
        lr = super(CappedExponentialDecay, self).__call__(step)
        return tf.maximum(lr, self.minimum_lr)

    def get_config(self):
        config = super(CappedExponentialDecay, self).get_config()
        config["minimum_lr"] = self.minimum_lr
        return config
