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

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import argparse
from datetime import datetime
from constructs.loggable import Loggable
from constructs.loops import Loops
from constructs.output import Output
from constructs.metrics import Metrics
from constructs.ema_weights import EMAWeights
import constructs.learning_rate as lr
import constructs.optimizer as opt
from models.SimpleMonolithic import SimpleMonolithic
from input.micro_pcb_input_pipeline import MicroPCB
from input.micro_pcb_input_pipeline2 import MicroPCBUseAllAndAugmentAll
import tensorflow as tf


def go(model_no, capsule_configuration, input_pipeline, run_name, end_step,
       lr_decay_steps, weights_file, data_dir, log_dir, batch_size,
       profile_batch_start, profile_batch_end, include_top5, image_size,
       train_rotations_to_use=None, train_perspectives_x_to_use=None,
       train_perspectives_y_to_use=None, test_rotations_to_use=None,
       test_perspectives_x_to_use=None, test_perspectives_y_to_use=None,
       generate_missing_rotations=False, generate_missing_perspectives=False):

    strategy = tf.distribute.MirroredStrategy()

    out = Output(log_dir, run_name, weights_file,
            profile_batch_start, profile_batch_end,
            include_top5=include_top5)

    if input_pipeline == 2:
        in_pipe = MicroPCBUseAllAndAugmentAll(
            out, data_dir, image_size, batch_size)
    else:
        in_pipe = MicroPCB(out, data_dir, image_size, batch_size,
            train_rotations_to_use, train_perspectives_x_to_use,
            train_perspectives_y_to_use, test_rotations_to_use,
            test_perspectives_x_to_use, test_perspectives_y_to_use,
            generate_missing_rotations, generate_missing_perspectives)

    with strategy.scope():
        out.print_msg("Building model...")

        if model_no == 2:
            model = SimpleMonolithic(in_pipe.get_class_count(), True,
                        caps_config=capsule_configuration)
        else:
            model = SimpleMonolithic(in_pipe.get_class_count(), False)

        learning_rate = lr.CappedExponentialDecay(
                            0.001, lr_decay_steps, 0.96, True, 1e-7)
        optimizer     = opt.Adam(learning_rate)

        metrics     = Metrics()
        ema_weights = EMAWeights(0.999, model.trainable_variables)
        loops       = Loops(in_pipe, out, strategy, model, optimizer,
                        learning_rate, metrics, ema_weights, batch_size,
                        weights_file)

        out.log_method_info(Loggable.get_this_method_info())
        out.log_loggables([out, in_pipe, model,
            learning_rate, optimizer, metrics, ema_weights, loops])

        epoch = 1
        while True:
            out.print_msg("Starting epoch {}...".format(epoch))
            loops.do_epoch(epoch)
            step_number = (epoch - 1) * in_pipe.get_train_steps()
            if step_number >= end_step:
                break
            epoch += 1


################################################################################
# Entry point
################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_no", default=1, type=int)
    p.add_argument("--capsule_configuration", default=6, type=int)
    p.add_argument("--input_pipeline", default=1, type=int)
    p.add_argument("--run_name", default=None)
    p.add_argument("--end_step", default=500000, type=int)
    p.add_argument("--lr_decay_steps", default=2200, type=int)
    p.add_argument("--weights_file", default=None)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--batch_size", default=2, type=int)
    p.add_argument("--profile_batch_start", default=None, type=int)
    p.add_argument("--profile_batch_end", default=None, type=int)
    p.add_argument("--include_top5", default=False, type=bool)
    p.add_argument("--image_size", default=299, type=int)
    p.add_argument("--trials", default=1, type=int)
    p.add_argument("--train_rotations_to_use", default=[1, 2, 3, 4, 5])
    p.add_argument("--train_perspectives_x_to_use", default=[1, 2, 3, 4, 5])
    p.add_argument("--train_perspectives_y_to_use", default=[1, 2, 3, 4, 5])
    p.add_argument("--test_rotations_to_use", default=[1, 2, 3, 4, 5])
    p.add_argument("--test_perspectives_x_to_use", default=[1, 2, 3, 4, 5])
    p.add_argument("--test_perspectives_y_to_use", default=[1, 2, 3, 4, 5])
    p.add_argument("--generate_missing_rotations", default=False, type=bool)
    p.add_argument("--generate_missing_perspectives", default=False, type=bool)
    a = p.parse_args()

    for i in range(a.trials):
        if a.run_name is None:
            rn = datetime.now().strftime("%Y%m%d%H%M%S")
        else:
            rn = a.run_name + ("" if a.trials <= 1 else "_" + str(i))

        go(model_no=a.model_no, capsule_configuration=a.capsule_configuration,
           input_pipeline=a.input_pipeline, run_name=rn, end_step=a.end_step,
           lr_decay_steps=a.lr_decay_steps, weights_file=a.weights_file,
           data_dir=a.data_dir, log_dir=a.log_dir, batch_size=a.batch_size,
           profile_batch_start=a.profile_batch_start,
           profile_batch_end=a.profile_batch_end, include_top5=a.include_top5,
           image_size=a.image_size,
           train_rotations_to_use=a.train_rotations_to_use,
           train_perspectives_x_to_use=a.train_perspectives_x_to_use,
           train_perspectives_y_to_use=a.train_perspectives_y_to_use,
           test_rotations_to_use=a.test_rotations_to_use,
           test_perspectives_x_to_use=a.test_perspectives_x_to_use,
           test_perspectives_y_to_use=a.test_perspectives_y_to_use,
           generate_missing_rotations=a.generate_missing_rotations,
           generate_missing_perspectives=a.generate_missing_perspectives)
