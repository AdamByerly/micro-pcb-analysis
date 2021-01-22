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

import tensorflow as tf
from models.model_base import ModelBase
from models.nn_ops import max_pool, conv_3x3
from models.nn_ops import caps_from_conv_xyz, hvc
from models.nn_ops import flatten, fc
from models.variable import get_conv_3x3_variable, get_hvc_variable
from models.variable import get_variable


class SimpleMonolithic(ModelBase):
    def __init__(self, count_classes, use_caps=True,
                 caps_config=6, name=None):
        ModelBase.__init__(self, name)
        with tf.name_scope("variables"):
            self._count_classes = count_classes
            self._use_caps      = use_caps
            self._caps_config   = caps_config

            self.W_conv1  = get_conv_3x3_variable("W_conv1", 3, 32)
            self.W_conv2  = get_conv_3x3_variable("W_conv2", 32, 32)
            self.W_conv3  = get_conv_3x3_variable("W_conv3", 32, 32)

            self.W_conv4  = get_conv_3x3_variable("W_conv4", 32, 64)
            self.W_conv5  = get_conv_3x3_variable("W_conv5", 64, 64)
            self.W_conv6  = get_conv_3x3_variable("W_conv6", 64, 64)

            self.W_conv7  = get_conv_3x3_variable("W_conv7", 64, 128)
            self.W_conv8  = get_conv_3x3_variable("W_conv8", 128, 128)
            self.W_conv9  = get_conv_3x3_variable("W_conv9", 128, 128)

            self.W_conv10 = get_conv_3x3_variable("W_conv10", 128, 256)
            self.W_conv11 = get_conv_3x3_variable("W_conv11", 256, 256)

            if self._use_caps:
                if caps_config == 1:
                    self.W_ocap = get_hvc_variable("W_ocap",
                                    81*32, self._count_classes, 8)
                elif caps_config == 2:
                    self.W_ocap = get_hvc_variable("W_ocap",
                                    81*16, self._count_classes, 16)
                elif caps_config == 3:
                    self.W_ocap = get_hvc_variable("W_ocap",
                                    81*8, self._count_classes, 32)
                elif caps_config == 4:
                    self.W_ocap = get_hvc_variable("W_ocap",
                                    81*4, self._count_classes, 64)
                elif caps_config == 5:
                    self.W_ocap = get_hvc_variable("W_ocap",
                                    81*2, self._count_classes, 128)
                else:
                    self.W_ocap = get_hvc_variable("W_ocap",
                                    81, self._count_classes, 256)
            else:
                self.W_fc   = get_variable([256*81,
                                self._count_classes], "W_fc")

    def forward(self, features, is_training):
        with tf.name_scope("forward"):
            x = conv_3x3("conv1", is_training,
                features, self.W_conv1, [1, 2, 2, 1])
            x = conv_3x3("conv2", is_training, x, self.W_conv2)
            x = conv_3x3("conv3", is_training, x, self.W_conv3)
            x = max_pool("pool1", x)

            x = conv_3x3("conv4", is_training, x, self.W_conv4)
            x = conv_3x3("conv5", is_training, x, self.W_conv5)
            x = conv_3x3("conv6", is_training, x, self.W_conv6)
            x = max_pool("pool2", x)

            x = conv_3x3("conv7", is_training, x, self.W_conv7)
            x = conv_3x3("conv8", is_training, x, self.W_conv8)
            x = conv_3x3("conv9", is_training, x, self.W_conv9)
            x = max_pool("pool3", x)

            x = conv_3x3("conv10", is_training, x, self.W_conv10)
            x = conv_3x3("conv11", is_training, x, self.W_conv11)

            if self._use_caps:
                if self._caps_config == 1:
                    x = caps_from_conv_xyz("pcap", x, 81*32, 8)
                elif self._caps_config == 2:
                    x = caps_from_conv_xyz("pcap", x, 81*16, 16)
                elif self._caps_config == 3:
                    x = caps_from_conv_xyz("pcap", x, 81*8, 32)
                elif self._caps_config == 4:
                    x = caps_from_conv_xyz("pcap", x, 81*4, 64)
                elif self._caps_config == 5:
                    x = caps_from_conv_xyz("pcap", x, 81*2, 128)
                else:
                    x = caps_from_conv_xyz("pcap", x, 81, 256)
                x = hvc("ocap", is_training, x, self.W_ocap)
            else:
                x = flatten("flatten", x)
                x = fc("fc", is_training, x, self.W_fc)

            with tf.name_scope("logits"):
                if self._use_caps:
                    logits = tf.reduce_sum(x, axis=2, name="logits")
                else:
                    logits = x
                return logits
