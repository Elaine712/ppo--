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
"""Deploy an Aero hand rotate-z ONNX policy in native MuJoCo."""

from __future__ import annotations

import argparse

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

from mujoco_playground._src.manipulation.aero_hand import aero_hand_constants
from mujoco_playground._src.manipulation.aero_hand.base import get_assets

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"

_SENSOR_TENDON_NAMES = (
    "len_if",
    "len_mf",
    "len_rf",
    "len_pf",
    "len_th1",
    "len_th2",
)
_SENSOR_JOINT_NAMES = ("len_th_abd",)
_DEFAULT_ACTION_SCALE = np.array(
    [0.02, 0.02, 0.02, 0.02, 0.5, 0.003, 0.012],
    dtype=np.float32,
)
_OBS_FRAME_SIZE = 16
_HISTORY_LEN = 4
_PHASE_FREQ = 0.6


class OnnxController:
  """ONNX controller for the TetherIA Aero Hand rotate-z task."""

  def __init__(
      self,
      policy_path: str,
      default_tendon: np.ndarray,
      ctrl_dt: float,
      n_substeps: int,
      action_scale: np.ndarray,
      ctrl_lower: np.ndarray,
      ctrl_upper: np.ndarray,
      max_action_delta: float | None = None,
      print_every: int = 0,
  ):
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )
    output_names = [output.name for output in self._policy.get_outputs()]
    self._output_names = (
        ["continuous_actions"]
        if "continuous_actions" in output_names
        else [output_names[0]]
    )

    self._default_tendon = default_tendon.astype(np.float32)
    self._action_scale = action_scale.astype(np.float32)
    self._ctrl_lower = ctrl_lower.astype(np.float32)
    self._ctrl_upper = ctrl_upper.astype(np.float32)
    self._last_action = np.zeros_like(default_tendon, dtype=np.float32)
    self._last_ctrl = self._default_tendon.copy()
    self._max_action_delta = max_action_delta

    self._counter = 0
    self._policy_steps = 0
    self._n_substeps = n_substeps
    self._print_every = print_every
    self._phase = 0.0
    self._phase_dt = 2 * np.pi * _PHASE_FREQ * ctrl_dt
    self._obs_history = np.zeros(
        _HISTORY_LEN * _OBS_FRAME_SIZE, dtype=np.float32
    )

  def get_obs(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    del model  # Unused.
    tendon_lengths = [data.sensor(name).data[0] for name in _SENSOR_TENDON_NAMES]
    joint_angles = [data.sensor(name).data[0] for name in _SENSOR_JOINT_NAMES]
    phase = np.array([np.cos(self._phase), np.sin(self._phase)])
    obs_frame = np.hstack([
        tendon_lengths,
        joint_angles,
        self._last_action,
        phase,
    ]).astype(np.float32)
    self._obs_history = np.roll(self._obs_history, obs_frame.size)
    self._obs_history[: obs_frame.size] = obs_frame
    return self._obs_history

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps != 0:
      return

    obs = self.get_obs(model, data)
    onnx_input = {"obs": obs.reshape(1, -1)}
    action = self._policy.run(self._output_names, onnx_input)[0][0].astype(
        np.float32
    )

    if self._max_action_delta is not None:
      delta = np.clip(
          action - self._last_action,
          -self._max_action_delta,
          self._max_action_delta,
      )
      action = self._last_action + delta

    ctrl = self._default_tendon + action * self._action_scale
    np.clip(ctrl, self._ctrl_lower, self._ctrl_upper, out=ctrl)
    data.ctrl[:] = ctrl

    self._last_action = action.copy()
    self._last_ctrl = ctrl.copy()
    self._phase = (self._phase + self._phase_dt + np.pi) % (2 * np.pi) - np.pi
    self._policy_steps += 1

    if self._print_every and self._policy_steps % self._print_every == 0:
      wz = data.sensor("cube_angvel").data[2]
      print(f"policy_step={self._policy_steps} wz={wz:.3f} ctrl={ctrl}")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run AeroCubeRotateZAxis ONNX policy in native MuJoCo."
  )
  parser.add_argument(
      "--policy_path",
      default=(_ONNX_DIR / "aero_rotate_z_policy.onnx").as_posix(),
      help="Path to an exported ONNX policy.",
  )
  parser.add_argument("--ctrl_dt", type=float, default=0.05)
  parser.add_argument("--sim_dt", type=float, default=0.01)
  parser.add_argument(
      "--action_scale_multiplier",
      type=float,
      default=1.0,
      help="Multiplier applied to the training action scale.",
  )
  parser.add_argument(
      "--max_action_delta",
      type=float,
      default=None,
      help="Optional per-control-step action slew-rate limit.",
  )
  parser.add_argument(
      "--init_wz",
      type=float,
      default=0.0,
      help="Initial cube z angular velocity in rad/s.",
  )
  parser.add_argument(
      "--print_every",
      type=int,
      default=0,
      help="Print cube z angular velocity every N policy steps.",
  )
  return parser.parse_args()


def load_callback(args: argparse.Namespace):
  mujoco.set_mjcb_control(None)

  if not epath.Path(args.policy_path).exists():
    raise FileNotFoundError(
        f"ONNX policy not found: {args.policy_path}. "
        "Run export_aero_rotate_z_onnx.py first, or pass --policy_path."
    )

  model = mujoco.MjModel.from_xml_path(
      aero_hand_constants.CUBE_XML.as_posix(),
      assets=get_assets(),
  )
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)

  model.opt.timestep = args.sim_dt
  n_substeps = int(round(args.ctrl_dt / args.sim_dt))
  if n_substeps < 1:
    raise ValueError("ctrl_dt must be greater than or equal to sim_dt.")

  cube_joint_id = model.joint("cube_freejoint").id
  cube_dofadr = model.jnt_dofadr[cube_joint_id]
  data.qvel[cube_dofadr + 5] = args.init_wz
  mujoco.mj_forward(model, data)

  policy = OnnxController(
      policy_path=args.policy_path,
      default_tendon=np.array(model.keyframe("home").ctrl),
      ctrl_dt=args.ctrl_dt,
      n_substeps=n_substeps,
      action_scale=_DEFAULT_ACTION_SCALE * args.action_scale_multiplier,
      ctrl_lower=model.actuator_ctrlrange[:, 0],
      ctrl_upper=model.actuator_ctrlrange[:, 1],
      max_action_delta=args.max_action_delta,
      print_every=args.print_every,
  )

  mujoco.set_mjcb_control(policy.get_control)
  return model, data


if __name__ == "__main__":
  _ARGS = _parse_args()
  viewer.launch(loader=lambda model=None, data=None: load_callback(_ARGS))
