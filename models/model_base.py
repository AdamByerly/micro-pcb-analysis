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
import tensorflow as tf
from constructs.loggable import Loggable
from models.variable import Variable


class ModelBase(tf.Module, Loggable, metaclass=abc.ABCMeta):
    def __init__(self, name=None):
        tf.Module.__init__(self, name=name)
        Loggable.__init__(self)
        self.all_vars = None

    def get_members(self):
        return {k: v for (k, v) in Loggable.get_members(self).items()
                if k != "all_vars" and
                   k.find("_self_unconditional_") < 0}

    @abc.abstractmethod
    def forward(self, features, is_training):
        pass

    # noinspection PyPropertyDefinition
    @property
    def trainable_variables(self):
        if self.all_vars is not None:
            return self.all_vars

        # get the trainable variables from tf.Module; this only finds
        # tf.Variables that are directly attributes of the Module
        base_vars      = super(ModelBase, self).trainable_variables

        # get all attributes of type Variable which themselves have 2
        # tf.Variable attributes
        variables      = [v for v in self.__dir__() if
                           v is not "trainable_variables"]
        variables      = [v for v in variables if
                           isinstance(getattr(self, v, None), Variable)]

        # get the trainable variables that are members of those
        variables_vars = [(getattr(self, v, None).variable,
                           getattr(self, v, None).bn_beta) for v in variables]

        self.all_vars = base_vars + sum(variables_vars, ())

        return self.all_vars
