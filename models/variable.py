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


# TODO: implement __str__()
class Variable(object):
    def __init__(self, initial_value, name):
        with tf.device("/device:CPU:0"):
            self.variable = tf.Variable(initial_value, name=name,
                aggregation=tf.VariableAggregation.MEAN)
            self.bn_beta = tf.Variable(tf.zeros(
                shape=[initial_value.shape[-1], ]), name=name+"_bn_beta",
                aggregation=tf.VariableAggregation.MEAN)
            self.bn_mean = tf.Variable(tf.zeros(
                shape=[initial_value.shape[-1], ]), name=name+"_bn_mean",
                trainable=False, aggregation=tf.VariableAggregation.MEAN)
            self.bn_variance = tf.Variable(tf.ones(
                shape=[initial_value.shape[-1], ]), name=name+"_bn_variance",
                trainable=False, aggregation=tf.VariableAggregation.MEAN)

    def get_trainable_variables(self):
        return self.variable, self.bn_beta


def get_variable(shape, name):
    return Variable(tf.random_normal_initializer()(shape=shape), name=name)


def get_conv_variable(name, input_size, filter_size_h, filter_size_w, filters):
    return get_variable(
        [filter_size_h, filter_size_w, input_size, filters], name)


def get_conv_3x3_variable(name, input_size, filters):
    return get_conv_variable(name, input_size, 3, 3, filters)


def get_hvc_variable(name, input_size, output_size, capsule_dimensions):
    return get_variable([output_size, input_size, capsule_dimensions], name)
