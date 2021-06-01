# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Utility functions to build losses, optimizers, EMA etc."""

import re

import tensorflow as tf
import enums

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu.ops import tpu_ops
# pylint: enable=g-direct-tensorflow-import

# LARS Optimizer: Scaling of learning rate to compute trust ratio
ETA_DEFAULT = 0.001

# Default global step used by create_train_op function
_USE_GLOBAL_STEP = 0


def maybe_add_warmup_to_lr(
    warmup_target, non_warmup_learning_rate, warmup_step_counter,
    warmup_epochs, steps_per_epoch):
  """Add warmup to the start of learning rate if warmup_steps is more than 0.

  Args:
    warmup_target: Learning rate at the end of the warmup period, before
      the decay kicks-in.
    non_warmup_learning_rate: The learning rate Tensor use after `warmup_steps`
      which also inherently depends on `global_step`. This could be a Tensor or
      python scalar.
    warmup_step_counter: The step Tensor to used to compute the learning rate.
      warmup_step_counter should take value 0 at the step warmup starts.
      Tensor has dtype tf.int32 or tf.int64.
    warmup_epochs: Tensor for number of epochs for which warmup runs. Dtype must
      be one of tf.int32, tf.int64 or tf.float32.
    steps_per_epoch: Tensor which defines the number of steps that are run for
      every epoch, with dtype

  Returns:
    Tensor which can be used as learning rate for the training process.
  """

  if warmup_epochs is not None:
    warmup_steps = tf.cast(warmup_epochs * steps_per_epoch,
                           warmup_step_counter.dtype)
    warmup_learning_rate = (
        warmup_target * tf.cast(warmup_step_counter, tf.float32) / tf.cast(
            warmup_steps, tf.float32))
    warmup_learning_rate = tf.cond(
        warmup_steps > 0,
        lambda: warmup_learning_rate,
        lambda: non_warmup_learning_rate)
    learning_rate = tf.cond(
        warmup_step_counter < warmup_steps,
        lambda: warmup_learning_rate,
        lambda: non_warmup_learning_rate)

  return learning_rate


def exponential_decay(initial_lr,
                      global_step,
                      total_epochs,
                      steps_per_epoch,
                      decay_rate=0.97,
                      epochs_per_decay=2.4):
  """Exponential decay learning rate decay.

  Args:
    initial_lr: Learning rate at the start of training if warmup is not applied.
    global_step: The global step Tensor to use for learning rate computation.
    total_epochs: Not used.
    steps_per_epoch: Number of training steps per epoch of training.
    decay_rate: The factor by which learning rate decays after every
      `epochs_per_decay` steps.
    epochs_per_decay: Scaling factor used to control the rate of decay.

  Returns:
    Tensor which can be used as learning rate for the training process.
  """
  del total_epochs
  epochs_per_decay = tf.convert_to_tensor(epochs_per_decay)
  decay_steps = tf.cast(
      steps_per_epoch, epochs_per_decay.dtype) * epochs_per_decay
  learning_rate = tf.train.exponential_decay(
      initial_lr, global_step, decay_steps, decay_rate, staircase=True)
  return learning_rate


def cosine_decay(initial_lr,
                 global_step,
                 total_epochs,
                 steps_per_epoch):
  r"""Cosine shaped learning rate decay.

  The learning rate multiplier varies as (1. + cos(x)) / 2. where x varies from
  [0, 2\pi] between step 0 and total_steps.

  Args:
    initial_lr: Learning rate at the start of training if warmup is not applied.
    global_step: The global step Tensor to use for learning rate computation.
    total_epochs: Total number of epochs over which the decay happens after
      which the learning rate is fixed at 0.
    steps_per_epoch: Number of training steps per epoch of training.

  Returns:
    Tensor which can be used as learning rate for the training
      process.
  """
  total_steps = tf.cast(
      steps_per_epoch * total_epochs,
      global_step.dtype)

  learning_rate = tf.train.cosine_decay(
      initial_lr, global_step, total_steps)

  return learning_rate


def piecewise_linear_decay(initial_lr,
                           global_step,
                           total_epochs,
                           steps_per_epoch,
                           boundary_epochs=(30, 60, 80, 90),
                           decay_rate=0.1):
  """Piece-wise linear learning rate schedule.

  Args:
    initial_lr: Learning rate at the start of training (without accounting for
      warmup).
    global_step: The global step to use for learning rate computation.
    total_epochs: Not used.
    steps_per_epoch: Number of training steps per epoch of training.
    boundary_epochs: Iterable of python ints containing epochs at which learning
      rate changes.
    decay_rate: At each `boundary_epoch`, `initial_lr` is decayed by an
      additional factor of `decay_rate`.

  Returns:
    Tensor which can be used as learning rate for the training process.
  """
  del total_epochs
  assert steps_per_epoch is not None
  boundaries = [tf.cast(steps_per_epoch * epoch, global_step.dtype)
                for epoch in boundary_epochs]
  rates = [initial_lr * decay_rate**n for n in range(len(boundary_epochs) + 1)]
  learning_rate = tf.compat.v1.train.piecewise_constant(global_step,
                                                        boundaries, rates)
  return learning_rate


def build_learning_rate_schedule(
    learning_rate,
    decay_type,
    warmup_start_epoch,
    max_learning_rate_epoch,
    decay_end_epoch,
    global_step,
    steps_per_epoch,
    **decay_type_specific_kwargs):
  """Build learning rate from base learning rate and other details.

  We note that warmup_start_epoch <= max_learning_rate_epoch < decay_end_epoch
  since the warmup happens at the start of learning rate schedule.

  Args:
    learning_rate: Learning rate for the model.
    decay_type: Name of the decay that should be applied to the learning rate.
    warmup_start_epoch: Epoch at which learning rate warmup starts.
    max_learning_rate_epoch: Epoch at which learning rate warmup ends and the
      decay kicks in.
    decay_end_epoch: Epoch at which learning rate decays ends, at which point
      learning rate becomes 0.
    global_step: The global step to use for learning rate computation.
    steps_per_epoch: Integer which defines the number of steps that are run for
      every epoch.
    **decay_type_specific_kwargs: Specific key-word arguments which are unique
      to a said `decay_type`.


  Returns:
    Scalar tensor which stores the learning rate at a given global step.
  """
  if decay_end_epoch == max_learning_rate_epoch:
    # This stage of training is 0 epochs long, so just return learning_rate and
    # avoid potential divide by 0 problems.
    if warmup_start_epoch < max_learning_rate_epoch:
      raise ValueError(
          'Cannot have warmup for a 0-step learning rate schedule.')

    return learning_rate

  assert warmup_start_epoch <= max_learning_rate_epoch
  assert max_learning_rate_epoch < decay_end_epoch

  max_learning_rate_epoch_tensor = tf.convert_to_tensor(max_learning_rate_epoch)
  warmup_start_epoch_tensor = tf.convert_to_tensor(
      warmup_start_epoch, max_learning_rate_epoch_tensor.dtype)
  decay_end_epoch_tensor = tf.convert_to_tensor(
      decay_end_epoch, max_learning_rate_epoch_tensor.dtype)
  steps_per_epoch_tensor = tf.cast(steps_per_epoch,
                                   max_learning_rate_epoch_tensor.dtype)

  # Learning rate decay kicks in starting max_learning_rate_epoch
  # Before max_learning_rate_epoch either there is a warmup or the learning rate
  # is set to the constant value of `initial_lr`.
  learning_rate_step = global_step - tf.cast(
      max_learning_rate_epoch_tensor * steps_per_epoch_tensor,
      global_step.dtype)

  def _no_decay_fn(initial_lr, *args, **kwargs):
    del args, kwargs
    return initial_lr

  decay_type_fn_map = {
      enums.DecayType.EXPONENTIAL: exponential_decay,
      enums.DecayType.COSINE: cosine_decay,
      enums.DecayType.PIECEWISE_LINEAR: piecewise_linear_decay,
      enums.DecayType.NO_DECAY: _no_decay_fn,
  }
  if decay_type not in decay_type_fn_map:
    raise ValueError(f'Unknown decay type {decay_type}')

  decayed_learning_rate = decay_type_fn_map[decay_type](
      initial_lr=learning_rate,
      global_step=learning_rate_step,
      total_epochs=decay_end_epoch_tensor - max_learning_rate_epoch_tensor,
      steps_per_epoch=steps_per_epoch,
      **decay_type_specific_kwargs)

  # The learning rate is set to 0 once global_step is more than total_steps.
  total_steps = tf.cast(
      steps_per_epoch_tensor * (
          decay_end_epoch_tensor - max_learning_rate_epoch_tensor),
      global_step.dtype)
  decayed_learning_rate = tf.cond(
      learning_rate_step <= total_steps,
      lambda: decayed_learning_rate,
      lambda: 0.0)

  warmup_step_counter = global_step - tf.cast(
      warmup_start_epoch_tensor * steps_per_epoch_tensor, global_step.dtype)
  return maybe_add_warmup_to_lr(
      learning_rate, decayed_learning_rate, warmup_step_counter,
      max_learning_rate_epoch - warmup_start_epoch_tensor,
      steps_per_epoch_tensor)


def stacked_multiview_image_channels_to_batch(images,
                                              data_format='channels_last'):
  """Split 2 views from the channel dim and concatenate back on the batch dim.

  Args:
    images: A 4-D batched image tensor, with 2 images stacked in the channel
      dimension.
    data_format: Either 'channels_first' or 'channels_last' to indicate whether
      `images` is formatted [N, 2C, H, W] or [N, H, W, 2C].

  Returns:
    Images reformated so that the extra views are now stacked in the batch
    dimension. If the input was [N, 2C, H, W] the output is [2N, C, H, W]. If
    the input was [N, H, W, 2C] the output is [2N, H, W, C].
  """
  with tf.name_scope('channels_to_batch'):
    if data_format == 'channels_first':
      images_a = images[:, :3, :, :]
      images_b = images[:, -3:, :, :]
    else:
      images_a = images[:, :, :, :3]
      images_b = images[:, :, :, -3:]
    return tf.concat([images_a, images_b], axis=0)


def stacked_multiview_embeddings_to_channel(embeddings):
  """Stack multiviewed embeddings in the channel dimension instead of batch.

  Args:
    embeddings: A 2D tensor of shape [2N, D].

  Returns:
    The embeddings reformatted to [N, 2D].
  """
  with tf.name_scope('batch_to_channels'):
    return tf.concat(tf.split(embeddings, 2, 0), 1)


def local_tpu_replica_id():
  """Returns the index of the current TPU replica."""
  num_tpu_replicas = tpu_function.get_tpu_context().number_of_shards
  if num_tpu_replicas is not None:
    # Need tf.control_dependencies(None) in order to make sure this is run
    # on CPU (not TPU)
    with tf.control_dependencies(None):
      return tpu_ops.tpu_replicated_input(
          list(range(num_tpu_replicas)), name='local_replica_id')
  else:
    # The non-TPU case.
    return 0


def cross_replica_concat(tensor):
  """A cross-replica concatenation of a single Tensor across TPU cores.

  Input tensor is assumed to have batch dimension as the first dimension. The
  concatenation is done along the batch dimension.

  Args:
    tensor: Input Tensor which should be concatenated across TPU cores.

  Returns:
    The concatenated Tensor with batch dimension multiplied by the number of
      TPU cores.
  """
  num_tpu_replicas = tpu_function.get_tpu_context().number_of_shards

  if num_tpu_replicas is not None:
    # Scattered tensor has shape [num_replicas, local_batch_size, ...]
    scattered_tensor = tf.scatter_nd(
        indices=[[local_tpu_replica_id()]],
        updates=[tensor],
        shape=[num_tpu_replicas] + tensor.shape.as_list())
    reduced_tensor = tf.tpu.cross_replica_sum(scattered_tensor)
    # Returned tensor has shape [num_replicas * local_batch_size, ...]
    return tf.reshape(reduced_tensor,
                      [-1] + scattered_tensor.shape.as_list()[2:])
  else:
    # This is a no op if not running on TPU
    return tensor


def estimator_mode_to_model_mode(estimator_mode):
  return {
      tf.estimator.ModeKeys.TRAIN: enums.ModelMode.TRAIN,
      tf.estimator.ModeKeys.EVAL: enums.ModelMode.EVAL,
      tf.estimator.ModeKeys.PREDICT: enums.ModelMode.INFERENCE,
  }[estimator_mode]
