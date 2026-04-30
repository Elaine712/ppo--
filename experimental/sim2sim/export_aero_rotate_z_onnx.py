# Copyright 2026 DeepMind Technologies Limited
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
# ==============================================================================
"""Export an AeroCubeRotateZAxis Brax PPO checkpoint to ONNX."""

from __future__ import annotations

import argparse
import pickle

from etils import epath
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tf2onnx

from brax.training.checkpoint import load as load_brax_checkpoint
from mujoco_playground import registry
from mujoco_playground.config import manipulation_params

_HERE = epath.Path(__file__).parent
_DEFAULT_OUTPUT_PATH = _HERE / "onnx" / "aero_rotate_z_policy.onnx"


class MLP(tf.keras.Model):
  """TensorFlow copy of the Brax PPO policy MLP used for ONNX export."""

  def __init__(
      self,
      layer_sizes,
      mean_std,
      activation=tf.nn.swish,
      kernel_init="lecun_uniform",
  ):
    super().__init__()
    self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
    self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
    self.mlp_block = tf.keras.Sequential(name="MLP_0")
    for i, size in enumerate(layer_sizes):
      dense_layer = layers.Dense(
          size,
          activation=activation,
          kernel_initializer=kernel_init,
          name=f"hidden_{i}",
          use_bias=True,
      )
      self.mlp_block.add(dense_layer)

    if self.mlp_block.layers:
      self.mlp_block.layers[-1].activation = None

  def call(self, inputs):
    if isinstance(inputs, list):
      inputs = inputs[0]
    inputs = (inputs - self.mean) / self.std
    logits = self.mlp_block(inputs)
    loc, _ = tf.split(logits, 2, axis=-1)
    return tf.tanh(loc)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Convert AeroCubeRotateZAxis checkpoint or params.pkl to ONNX."
  )
  parser.add_argument(
      "--checkpoint_path",
      required=True,
      help="Path to a Brax checkpoint directory or a params.pkl file.",
  )
  parser.add_argument(
      "--output_path",
      default=_DEFAULT_OUTPUT_PATH.as_posix(),
      help="Output ONNX path.",
  )
  parser.add_argument(
      "--env_name",
      default="AeroCubeRotateZAxis",
      help="Environment name used for the checkpoint.",
  )
  return parser.parse_args()


def _tree_get(obj, key):
  if hasattr(obj, key):
    return getattr(obj, key)
  return obj[key]


def _load_params(path: str):
  path_obj = epath.Path(path)
  if path_obj.name.endswith(".pkl"):
    with path_obj.open("rb") as f:
      data = pickle.load(f)
    return data["normalizer_params"], data["policy_params"]

  params = load_brax_checkpoint(path)
  if isinstance(params, dict) and "normalizer_params" in params:
    return params["normalizer_params"], params["policy_params"]
  if isinstance(params, (tuple, list)) and len(params) >= 2:
    return params[0], params[1]
  raise ValueError(
      "Unsupported checkpoint format. Expected a Brax checkpoint or params.pkl."
  )


def _as_numpy(value):
  return np.asarray(value, dtype=np.float32)


def _transfer_weights(policy_params, tf_model: tf.keras.Model) -> None:
  jax_params = policy_params["params"]
  mlp = tf_model.get_layer("MLP_0")
  for layer_name, layer_params in jax_params.items():
    tf_layer = mlp.get_layer(name=layer_name)
    kernel = np.asarray(layer_params["kernel"])
    bias = np.asarray(layer_params["bias"])
    tf_layer.set_weights([kernel, bias])


def main() -> None:
  args = _parse_args()
  ppo_params = manipulation_params.brax_ppo_config(args.env_name)
  env = registry.load(args.env_name)

  obs_size = env.observation_size
  obs_dim = obs_size["state"][0] if isinstance(obs_size, dict) else obs_size[0]
  act_size = env.action_size

  normalizer_params, policy_params = _load_params(args.checkpoint_path)
  mean = _as_numpy(_tree_get(normalizer_params, "mean")["state"])
  std = _as_numpy(_tree_get(normalizer_params, "std")["state"])

  tf_policy_network = MLP(
      layer_sizes=list(ppo_params.network_factory.policy_hidden_layer_sizes)
      + [act_size * 2],
      mean_std=(mean, std),
  )
  tf_policy_network(tf.zeros((1, obs_dim), dtype=tf.float32))
  _transfer_weights(policy_params, tf_policy_network)

  output_path = epath.Path(args.output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  spec = [
      tf.TensorSpec(shape=(1, obs_dim), dtype=tf.float32, name="obs"),
  ]
  try:
    tf_policy_network.output_names = ["continuous_actions"]
  except AttributeError:
    pass

  tf2onnx.convert.from_keras(
      tf_policy_network,
      input_signature=spec,
      opset=11,
      output_path=output_path.as_posix(),
  )
  print(f"Exported ONNX policy to: {output_path}")


if __name__ == "__main__":
  main()
