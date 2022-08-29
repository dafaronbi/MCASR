"""Library of loss functions."""

import functools
from typing import Dict, Text

import crepe
from ddsp import core
from ddsp import dags
from ddsp import spectral_ops
from ddsp.core import hz_to_midi
from ddsp.core import safe_divide
from ddsp.core import tf_float32

import gin
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfkl = tf.keras.layers

# Define Types.
TensorDict = Dict[Text, tf.Tensor]


# ---------------------- Base Classes ------------------------------------------
class Loss(tfkl.Layer):
  """Base class. Duck typing: Losses just must implement get_losses_dict()."""

  def get_losses_dict(self, *args, **kwargs):
    """Returns a dictionary of losses for the model."""
    loss = self(*args, **kwargs)
    return {self.name: loss}


@gin.register
class LossGroup(dags.DAGLayer):
  """Compute a group of loss layers on an outputs dictionary."""

  def __init__(self, dag: dags.DAG, **kwarg_losses):
    """Constructor, completely configurable via gin.
    Args:
      dag: A list of loss names/instances, with keys to extract call() inputs
        from a dictionary, ex:
        ['module', ['input_key', ...]]
        'module': Loss module instance or string name of module. For example,
          'spectral_loss' would access the attribute `loss_group.spectral_loss`.
        'input_key': List of strings, nested keys in dictionary of dag outputs.
      **kwarg_losses: Losses to add to LossGroup. Each kwarg that is a Loss will
        be added as a property of the layer, so that it will be accessible as
        `loss_group.kwarg`. Also, other keras kwargs such as 'name' are split
        off before adding modules.
    """
    super().__init__(dag, **kwarg_losses)
    self.loss_names = self.module_names

  @property
  def losses(self):
    """Loss getter."""
    return [getattr(self, name) for name in self.loss_names]

  def call(self, outputs: TensorDict, **kwargs) -> TensorDict:
    """Get a dictionary of loss values from all the losses.
    Args:
      outputs: A dictionary of model output tensors to feed into the losses.
      **kwargs: Other kwargs for all the modules in the dag.
    Returns:
      A flat dictionary of losses {name: scalar}.
    """
    dag_outputs = super().call(outputs, **kwargs)
    loss_outputs = {}
    for k in self.loss_names:
      loss_outputs.update(dag_outputs[k])
    return loss_outputs

  def get_losses_dict(self, outputs, **kwargs):
    """Returns a dictionary of losses for the model, alias __call__."""
    return self(outputs, **kwargs)

def mean_difference(target, value, loss_type='L1', weights=None):
  """Common loss functions.
  Args:
    target: Target tensor.
    value: Value tensor.
    loss_type: One of 'L1', 'L2', or 'COSINE'.
    weights: A weighting mask for the per-element differences.
  Returns:
    The average loss.
  Raises:
    ValueError: If loss_type is not an allowed value.
  """
  difference = target - value
  weights = 1.0 if weights is None else weights
  loss_type = loss_type.upper()
  if loss_type == 'L1':
    return tf.reduce_mean(tf.abs(difference * weights))
  elif loss_type == 'L2':
    return tf.reduce_mean(difference**2 * weights)
  elif loss_type == 'COSINE':
    return tf.losses.cosine_distance(target, value, weights=weights, axis=-1)
  else:
    raise ValueError('Loss type ({}), must be '
                     '"L1", "L2", or "COSINE"'.format(loss_type))


class SpectralLoss(Loss):
  """Multi-scale spectrogram loss.
  This loss is the bread-and-butter of comparing two audio signals. It offers
  a range of options to compare spectrograms, many of which are redunant, but
  emphasize different aspects of the signal. By far, the most common comparisons
  are magnitudes (mag_weight) and log magnitudes (logmag_weight).
  """

  def __init__(self,
               fft_sizes=(2048, 1024, 512, 256, 128, 64),
               loss_type='L1',
               mag_weight=1.0,
               delta_time_weight=0.0,
               delta_freq_weight=0.0,
               cumsum_freq_weight=0.0,
               logmag_weight=0.0,
               loudness_weight=0.0,
               name='spectral_loss'):
    """Constructor, set loss weights of various components.
    Args:
      fft_sizes: Compare spectrograms at each of this list of fft sizes. Each
        spectrogram has a time-frequency resolution trade-off based on fft size,
        so comparing multiple scales allows multiple resolutions.
      loss_type: One of 'L1', 'L2', or 'COSINE'.
      mag_weight: Weight to compare linear magnitudes of spectrograms. Core
        audio similarity loss. More sensitive to peak magnitudes than log
        magnitudes.
      delta_time_weight: Weight to compare the first finite difference of
        spectrograms in time. Emphasizes changes of magnitude in time, such as
        at transients.
      delta_freq_weight: Weight to compare the first finite difference of
        spectrograms in frequency. Emphasizes changes of magnitude in frequency,
        such as at the boundaries of a stack of harmonics.
      cumsum_freq_weight: Weight to compare the cumulative sum of spectrograms
        across frequency for each slice in time. Similar to a 1-D Wasserstein
        loss, this hopefully provides a non-vanishing gradient to push two
        non-overlapping sinusoids towards eachother.
      logmag_weight: Weight to compare log magnitudes of spectrograms. Core
        audio similarity loss. More sensitive to quiet magnitudes than linear
        magnitudes.
      loudness_weight: Weight to compare the overall perceptual loudness of two
        signals. Very high-level loss signal that is a subset of mag and
        logmag losses.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.fft_sizes = fft_sizes
    self.loss_type = loss_type
    self.mag_weight = mag_weight
    self.delta_time_weight = delta_time_weight
    self.delta_freq_weight = delta_freq_weight
    self.cumsum_freq_weight = cumsum_freq_weight
    self.logmag_weight = logmag_weight
    self.loudness_weight = loudness_weight

    self.spectrogram_ops = []
    for size in self.fft_sizes:
      spectrogram_op = functools.partial(spectral_ops.compute_mag, size=size)
      self.spectrogram_ops.append(spectrogram_op)

  def call(self, target_audio, audio, weights=None):
    loss = 0.0

    diff = core.diff
    cumsum = tf.math.cumsum

    # Compute loss for each fft size.
    for loss_op in self.spectrogram_ops:
      target_mag = loss_op(target_audio)
      value_mag = loss_op(audio)

      # Add magnitude loss.
      if self.mag_weight > 0:
        loss += self.mag_weight * mean_difference(
            target_mag, value_mag, self.loss_type, weights=weights)

      if self.delta_time_weight > 0:
        target = diff(target_mag, axis=1)
        value = diff(value_mag, axis=1)
        loss += self.delta_time_weight * mean_difference(
            target, value, self.loss_type, weights=weights)

      if self.delta_freq_weight > 0:
        target = diff(target_mag, axis=2)
        value = diff(value_mag, axis=2)
        loss += self.delta_freq_weight * mean_difference(
            target, value, self.loss_type, weights=weights)

      # TODO(kyriacos) normalize cumulative spectrogram
      if self.cumsum_freq_weight > 0:
        target = cumsum(target_mag, axis=2)
        value = cumsum(value_mag, axis=2)
        loss += self.cumsum_freq_weight * mean_difference(
            target, value, self.loss_type, weights=weights)

      # Add logmagnitude loss, reusing spectrogram.
      if self.logmag_weight > 0:
        target = spectral_ops.safe_log(target_mag)
        value = spectral_ops.safe_log(value_mag)
        loss += self.logmag_weight * mean_difference(
            target, value, self.loss_type, weights=weights)

    if self.loudness_weight > 0:
      target = spectral_ops.compute_loudness(target_audio, n_fft=2048,
                                             use_tf=True)
      value = spectral_ops.compute_loudness(audio, n_fft=2048, use_tf=True)
      loss += self.loudness_weight * mean_difference(
          target, value, self.loss_type, weights=weights)

    return loss

