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
from datetime import datetime
from constructs.loggable import Loggable
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2


class Output(Loggable):
    def __init__(self, log_dir, run_name, prior_weights_file=None,
            profile_batch_start=None, profile_batch_end=None,
            console_log_interval=1, include_top5=False):
        Loggable.__init__(self)
        self._log_dir              = os.path.join(log_dir, run_name)
        self._run_name             = run_name
        self._latest_weights_file  = prior_weights_file
        self._pbs                  = profile_batch_start
        self._pbe                  = profile_batch_start \
                                        if profile_batch_end is None \
                                        else profile_batch_end
        self._console_log_interval = console_log_interval
        self._summary_writer       = tf.summary.create_file_writer(
                                        self._log_dir)
        self._profile_started      = False
        self._profile_finished     = False
        self._ckpt_latest          = None
        self._ckpt_latest_mgr      = None
        self._ckpt_best_top1       = None
        self._ckpt_best_top1_mgr   = None
        self._best_top1_accuracy   = 0
        self._include_top5         = include_top5
        self._ckpt_best_top5       = None
        self._ckpt_best_top5_mgr   = None
        self._best_top5_accuracy   = 0

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)

    def get_summary_writer(self):
        return self._summary_writer

    def train_step_begin(self, step):
        if self._pbs is not None and step + 1 >= self._pbs \
                and self._profile_started is not True \
                and self._profile_finished is not True:
            tf.profiler.experimental.start(self._log_dir)
            self.print_msg("Profiling started...")
            self._profile_started = True

    def train_step_end(self, epoch, step, loss, lr, number_of_training_steps):
        self._log_metrics(epoch=epoch, loss=loss, lr=lr, step=step,
            steps_per_epoch=number_of_training_steps, is_test=False)

        if self._profile_started and step + 1 >= self._pbe \
                and not self._profile_finished:
            tf.profiler.experimental.stop()
            self.print_msg("Profiling finished...")
            self._profile_finished = True

    # noinspection PyUnusedLocal
    def train_end(self, model, optimizer, ema_weights, epoch):
        if self._ckpt_latest is None:
            self._ckpt_latest = tf.train.Checkpoint(
                vars=model.trainable_variables, optimizer=optimizer)
            # TODO: Add ema_weights.ema_object to the checkpoint.
            #       Currently, tf doesn't seem to support this.
            #       See https://github.com/tensorflow/tensorflow/issues/38452
            self._ckpt_latest_mgr = tf.train.CheckpointManager(
                self._ckpt_latest, self._log_dir,
                max_to_keep=2, checkpoint_name='latest')

        self._ckpt_latest_mgr.save(epoch)

    def validation_step_end(self, step, number_of_validation_steps):
        if step % self._console_log_interval == 0:
            self.print_msg("Validating (step {}/{})...".format(
                step, number_of_validation_steps), True)

    # noinspection PyUnusedLocal
    def validation_end(self, model, optimizer, ema_weights,
            epoch, loss, top1_accuracy, top5_accuracy=None):
        self._log_metrics(epoch=epoch, loss=loss, lr=0,
            is_test=True, top1_accuracy=top1_accuracy)

        if self._ckpt_best_top1 is None:
            self._ckpt_best_top1 = tf.train.Checkpoint(
                vars=model.trainable_variables, optimizer=optimizer)
            # TODO: Add ema_weights.ema_object to the checkpoint.
            #       Currently, tf doesn't seem to support this.
            #       See https://github.com/tensorflow/tensorflow/issues/38452
            self._ckpt_best_top1_mgr = tf.train.CheckpointManager(
                self._ckpt_best_top1, self._log_dir,
                max_to_keep=2, checkpoint_name='best_top1')

        if top1_accuracy >= self._best_top1_accuracy:
            self._best_top1_accuracy = top1_accuracy
            self._ckpt_best_top1_mgr.save(epoch)

        if self._include_top5:
            if self._ckpt_best_top5 is None:
                self._ckpt_best_top5 = tf.train.Checkpoint(
                    vars=model.trainable_variables, optimizer=optimizer)
                # TODO: ditto above
                self._ckpt_best_top5_mgr = tf.train.CheckpointManager(
                    self._ckpt_best_top5, self._log_dir,
                    max_to_keep=2, checkpoint_name='best_top5')

            if top5_accuracy >= self._best_top5_accuracy:
                self._best_top1_accuracy = top5_accuracy
                self._ckpt_best_top5_mgr.save(epoch)

    def _log_metrics(self, epoch, loss, lr=0, step=0, steps_per_epoch=0,
            is_test=False, top1_accuracy=None, top5_accuracy=None):
        with self._summary_writer.as_default():
            if is_test:
                tf.summary.scalar("Test/Loss", loss, epoch)
                tf.summary.scalar("Test/Top-1 Accuracy", top1_accuracy, epoch)
                self.print_msg("[TEST] - Epoch {}".format(epoch))
                self.print_msg("top1: {}".format(top1_accuracy), False)
                if self._include_top5:
                    tf.summary.scalar("Test/Top-5 Accuracy",
                        top5_accuracy, epoch)
                    self.print_msg("top5: {}".format(top5_accuracy), False)
            else:
                step_number = (epoch - 1) * steps_per_epoch + step
                tf.summary.scalar("Train/Loss", loss, step_number)
                tf.summary.scalar("Train/LR", lr, step_number)
                self.print_msg("[TRAIN] - Epoch {}, Step {}"
                    .format(epoch, step))

            self.print_msg("loss: {}".format(loss), False)

    def log_method_info(self, method_info):
        with self._summary_writer.as_default():
            tf.summary.text("Logging/" + method_info[0],
                tf.convert_to_tensor(method_info[1]), 0)

    def log_graph_of_func(self, func, features, labels):
        with self._summary_writer.as_default():
            func_graph = func.get_concrete_function(features, labels).graph
            summary_ops_v2.graph(func_graph, step=0)

    def log_loggables(self, loggables):
        with self._summary_writer.as_default():
            for loggable in loggables:
                file, data = loggable.get_log_data()
                tf.summary.text("Logging/"+file, tf.convert_to_tensor(data), 0)

    @staticmethod
    def print_msg(msg, put_time=True):
        t_str = "                     "
        if put_time:
            t_str = datetime.now().strftime("%Y%m%d %H:%M:%S.%f")[:-3]
        print("{} - {}".format(t_str, msg))
