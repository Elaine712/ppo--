# Copyright 2025 TetherIA Inc.
# Copyright 2025 DeepMind Technologies Limited
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

"""Rotate-z with TetherIA Aero Hand Open."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aero_hand import aero_hand_constants as consts
from mujoco_playground._src.manipulation.aero_hand import base as aero_hand_base


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,  # 20 Hz policy control.
        sim_dt=0.01,  # 100 Hz physics simulation.
        action_scale=[0.02, 0.02, 0.02, 0.02, 0.5, 0.003, 0.012],
        target_angvel=0.35,
        min_angvel=0.08,
        overspeed_cost=1.0,
        phase_freq=0.6,
        action_repeat=1,
        episode_length=500,
        early_termination=True,
        history_len=4,
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.05,
                tendon_length=0.005,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                angvel=2.0,
                stall=-0.5,
                pos_error=-0.5,
                fingertip_dist=-0.15,
                fingertip_max=-0.2,
                fingertip_balance=-0.5,
                action_balance=-0.05,
                linvel=-0.08,
                pose=-0.02,
                torques=0.0,
                energy=0.0,
                termination=-50.0,
                action_rate=-0.05,
            )
        ),
    )


class CubeRotateZAxis(aero_hand_base.AeroHandEnv):
    """Rotate a cube slowly around the z-axis without dropping it."""

    def __init__(
            self,
            config: config_dict.ConfigDict = default_config(),
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.CUBE_XML.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)

        self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
        self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._cube_geom_id = self._mj_model.geom("cube").id

        home_key = self._mj_model.keyframe("home")
        self._init_q = jp.array(home_key.qpos)
        self._default_pose = self._init_q[self._hand_qids]
        self._lowers, self._uppers = self.mj_model.jnt_range[self._hand_qids].T

        self._init_tendon = jp.array(home_key.ctrl)
        self._default_tendon = self._init_tendon

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Randomize hand qpos and qvel.
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        q_hand = jp.clip(
            self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
            self._lowers,
            self._uppers,
        )
        v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

        # Randomize cube qpos and qvel.
        rng, p_rng, quat_rng = jax.random.split(rng, 3)
        start_pos = jp.array([0.1, 0.0, 0.05]) + jax.random.uniform(
            p_rng, (3,), minval=-0.01, maxval=0.01
        )
        start_quat = aero_hand_base.uniform_quat(quat_rng)
        q_cube = jp.array([*start_pos, *start_quat])
        # v_cube = jp.zeros(6)
        rng, wz_rng = jax.random.split(rng)

        # Start near rest for the slow, stable rotation task.
        init_wz = jax.random.uniform(wz_rng, minval=0.0, maxval=0.2)
        v_cube = jp.array([0.0, 0.0, 0.0, 0.0, 0.0, init_wz])
        qpos = jp.concatenate([q_hand, q_cube])
        qvel = jp.concatenate([v_hand, v_cube])
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=self._default_tendon,  # Change: only use the control tendons
            mocap_pos=jp.array([-100, -100, -100]),  # Hide goal for this task.
        )

        rng, phase_rng = jax.random.split(rng)
        info = {
            "rng": rng,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": data.ctrl,
            "last_cube_angvel": jp.zeros(3),
            "phase": jax.random.uniform(phase_rng, minval=-jp.pi, maxval=jp.pi),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        # 6 tendon sensors + 1 joint sensor + 7 previous actions + 2 phase values.
        obs_history = jp.zeros(self._config.history_len * 16)
        obs = self._get_obs(data, info, obs_history)
        reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        action_scale_custom = jp.array(self._config.action_scale, dtype=jp.float32)
        motor_targets = self._default_tendon + action * action_scale_custom
        # NOTE: no clipping.
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )
        state.info["motor_targets"] = motor_targets
        phase_tp1 = state.info["phase"] + 2 * jp.pi * self._config.phase_freq * self.dt
        state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi

        obs = self._get_obs(data, state.info, state.obs["state"])
        done = self._get_termination(data)

        rewards = self._get_reward(data, action, state.info, state.metrics, done)
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = sum(rewards.values()) * self.dt  # pylint: disable=redefined-outer-name

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["last_cube_angvel"] = self.get_cube_angvel(data)
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_cube_position(data)[2] < -0.05
        return fall_termination

    def _get_obs(
            self, data: mjx.Data, info: dict[str, Any], obs_history: jax.Array
    ) -> Dict[str, jax.Array]:

        info["rng"], noise_rng = jax.random.split(info["rng"])

        # ------- tendon length sensor -------
        tendon_lengths = jp.zeros(
            (len(consts.SENSOR_TENDON_NAMES),), dtype=jp.float32
        )
        for idx, name in enumerate(consts.SENSOR_TENDON_NAMES):
            v = mjx_env.get_sensor_data(self.mj_model, data, name)
            v = jp.ravel(v)[0]
            tendon_lengths = tendon_lengths.at[idx].set(v)

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_tendon_lengths = (
                tendon_lengths
                + (2 * jax.random.uniform(noise_rng, shape=tendon_lengths.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.tendon_length
        )

        # ------- joint angle sensor -------
        joint_angles = jp.zeros((len(consts.SENSOR_JOINT_NAMES),), dtype=jp.float32)
        for idx, name in enumerate(consts.SENSOR_JOINT_NAMES):
            v = mjx_env.get_sensor_data(self.mj_model, data, name)
            v = jp.ravel(v)[0]
            joint_angles = joint_angles.at[idx].set(v)

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
                joint_angles
                + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.joint_pos
        )

        state = jp.concatenate([
            noisy_tendon_lengths,
            noisy_joint_angles,
            info["last_act"],
            jp.array([jp.cos(info["phase"]), jp.sin(info["phase"])]),
        ])

        joint_angles = data.qpos[self._hand_qids]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        obs_history = jp.roll(obs_history, state.size)
        obs_history = obs_history.at[: state.size].set(state)

        cube_pos = self.get_cube_position(data)
        palm_pos = self.get_palm_position(data)
        cube_pos_error = palm_pos - cube_pos
        cube_quat = self.get_cube_orientation(data)
        cube_angvel = self.get_cube_angvel(data)
        cube_linvel = self.get_cube_linvel(data)
        fingertip_positions = self.get_fingertip_positions(data)
        joint_torques = data.actuator_force

        privileged_state = jp.concatenate([
            state,
            joint_angles,
            data.qvel[self._hand_dqids],
            joint_torques,
            fingertip_positions,
            cube_pos_error,
            cube_quat,
            cube_angvel,
            cube_linvel,
        ])

        return {
            "state": obs_history,
            "privileged_state": privileged_state,
        }

    def _get_reward(
            self,
            data: mjx.Data,
            action: jax.Array,
            info: dict[str, Any],
            metrics: dict[str, Any],
            done: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.
        cube_pos = self.get_cube_position(data)
        palm_pos = self.get_palm_position(data)
        cube_pos_error = palm_pos - cube_pos
        cube_angvel = self.get_cube_angvel(data)
        cube_linvel = self.get_cube_linvel(data)

        fingertip_positions = self.get_fingertip_positions(data)
        fingertip_positions_3d = jp.reshape(fingertip_positions, (5, 3))
        cube_pos_rel = cube_pos - palm_pos
        fingertip_dists = jp.linalg.norm(fingertip_positions_3d - cube_pos_rel, axis=-1)
        fingertip_dist = jp.mean(fingertip_dists)
        fingertip_max = jp.max(fingertip_dists)
        fingertip_balance = jp.var(fingertip_dists)

        return {
            "angvel": self._reward_angvel(cube_angvel, cube_pos_error),
            "stall": self._cost_stall(cube_angvel),
            "pos_error": self._cost_pos_error(cube_pos_error),
            "fingertip_dist": fingertip_dist,
            "fingertip_max": fingertip_max,
            "fingertip_balance": fingertip_balance,
            "linvel": self._cost_linvel(cube_linvel),
            "termination": done,
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info.get("last_last_act", info["last_act"])
            ),
            "action_balance": self._cost_action_balance(action),
            "pose": self._cost_pose(data.qpos[self._hand_qids]),
            "torques": self._cost_torques(data.actuator_force),
            "energy": self._cost_energy(
                data.qvel[self._hand_dqids], data.qfrc_actuator[self._hand_dqids]
            ),
        }

    def _cost_pose_strict(self, joint_angles: jax.Array) -> jax.Array:
        diff = joint_angles - self._default_pose
        return jp.sum(jp.power(diff, 4))

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torques))

    def _cost_energy(
            self, qvel: jax.Array, qfrc_actuator: jax.Array
    ) -> jax.Array:
        return jp.sum(
            jp.abs(qvel) * jp.abs(qfrc_actuator)
        )  # Change: only use the control joints

    def _cost_linvel(self, cube_linvel: jax.Array) -> jax.Array:
        return jp.linalg.norm(cube_linvel, ord=1, axis=-1)

    def _reward_angvel(
            self, cube_angvel: jax.Array, cube_pos_error: jax.Array
    ) -> jax.Array:
        del cube_pos_error  # Unused.

        wz = cube_angvel[2]
        target_wz = self._config.target_angvel
        forward_reward = jp.clip(wz / target_wz, a_min=0.0, a_max=1.0)
        overspeed = jp.maximum(wz - target_wz, 0.0)
        reverse = jp.maximum(-wz, 0.0)
        return (
            forward_reward
            - self._config.overspeed_cost * jp.square(overspeed)
            - 0.5 * jp.square(reverse)
        )

    def _cost_stall(self, cube_angvel: jax.Array) -> jax.Array:
        wz = cube_angvel[2]
        return jp.clip((self._config.min_angvel - wz) / self._config.min_angvel, 0.0, 1.0)

    def _cost_pos_error(self, cube_pos_error: jax.Array) -> jax.Array:

        return jp.linalg.norm(cube_pos_error)

    def _cost_action_rate(
            self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        del last_last_act  # Unused.
        return jp.sum(jp.square(act - last_act))

    def _cost_action_balance(self, act: jax.Array) -> jax.Array:
        finger_actions = act[:4]
        return jp.var(finger_actions)

    def _cost_pose(self, joint_angles: jax.Array) -> jax.Array:
        return jp.sum(jp.square(joint_angles - self._default_pose))


def domain_randomize(model: mjx.Model, rng: jax.Array):
    mj_model = CubeRotateZAxis().mj_model
    cube_geom_id = mj_model.geom("cube").id
    cube_body_id = mj_model.body("cube").id
    hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
    hand_body_names = [
        "palm",
        "right_index_f_link",
        "right_index_proximal_link",
        "right_index_middle_link",
        "right_index_distal_link",
        "right_middle_f_link",
        "right_middle_proximal_link",
        "right_middle_middle_link",
        "right_middle_distal_link",
        "right_ring_f_link",
        "right_ring_proximal_link",
        "right_ring_middle_link",
        "right_ring_distal_link",
        "right_pinky_f_link",
        "right_pinky_proximal_link",
        "right_pinky_middle_link",
        "right_pinky_distal_link",
        "right_t_link",
        "right_thumb_mcp_link",
        "right_thumb_proximal_link",
        "right_thumb_distal_link",
    ]
    hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
    fingertip_geoms = ["if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip"]
    fingertip_geom_ids = [mj_model.geom(g).id for g in fingertip_geoms]

    @jax.vmap
    def rand(rng):
        # Cube friction: =U(0.1, 0.5).
        rng, key = jax.random.split(rng)
        cube_friction = jax.random.uniform(key, (1,), minval=0.1, maxval=0.5)
        geom_friction = model.geom_friction.at[
                        cube_geom_id: cube_geom_id + 1, 0
                        ].set(cube_friction)

        # Fingertip friction: =U(0.5, 1.0).
        fingertip_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=1.0)
        geom_friction = model.geom_friction.at[fingertip_geom_ids, 0].set(
            fingertip_friction
        )

        # Scale cube mass: *U(0.8, 1.2).
        rng, key1, key2 = jax.random.split(rng, 3)
        dmass = jax.random.uniform(key1, minval=0.8, maxval=1.2)
        cube_mass = model.body_mass[cube_body_id]
        body_mass = model.body_mass.at[cube_body_id].set(cube_mass * dmass)
        body_inertia = model.body_inertia.at[cube_body_id].set(
            model.body_inertia[cube_body_id] * dmass
        )
        dpos = jax.random.uniform(key2, (3,), minval=-5e-3, maxval=5e-3)
        body_ipos = model.body_ipos.at[cube_body_id].set(
            model.body_ipos[cube_body_id] + dpos
        )

        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[hand_qids].set(
            qpos0[hand_qids]
            + jax.random.uniform(key, shape=(16,), minval=-0.05, maxval=0.05)
        )

        # Scale static friction: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
            key, shape=(16,), minval=0.5, maxval=2.0
        )
        dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[hand_qids] * jax.random.uniform(
            key, shape=(16,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[hand_qids].set(armature)

        # Scale all link masses: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1
        )
        body_mass = model.body_mass.at[hand_body_ids].set(
            model.body_mass[hand_body_ids] * dmass
        )

        # Joint stiffness: *U(0.8, 1.2).
        rng, key = jax.random.split(rng)
        kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
            key, (model.nu,), minval=0.8, maxval=1.2
        )
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

        # Joint damping: *U(0.8, 1.2).
        rng, key = jax.random.split(rng)
        kd = model.dof_damping[hand_qids] * jax.random.uniform(
            key, (16,), minval=0.8, maxval=1.2
        )
        dof_damping = model.dof_damping.at[hand_qids].set(kd)

        return (
            geom_friction,
            body_mass,
            body_inertia,
            body_ipos,
            qpos0,
            dof_frictionloss,
            dof_armature,
            dof_damping,
            actuator_gainprm,
            actuator_biasprm,
        )

    (
        geom_friction,
        body_mass,
        body_inertia,
        body_ipos,
        qpos0,
        dof_frictionloss,
        dof_armature,
        dof_damping,
        actuator_gainprm,
        actuator_biasprm,
    ) = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "body_mass": 0,
        "body_inertia": 0,
        "body_ipos": 0,
        "qpos0": 0,
        "dof_frictionloss": 0,
        "dof_armature": 0,
        "dof_damping": 0,
        "actuator_gainprm": 0,
        "actuator_biasprm": 0,
    })

    model = model.tree_replace({
        "geom_friction": geom_friction,
        "body_mass": body_mass,
        "body_inertia": body_inertia,
        "body_ipos": body_ipos,
        "qpos0": qpos0,
        "dof_frictionloss": dof_frictionloss,
        "dof_armature": dof_armature,
        "dof_damping": dof_damping,
        "actuator_gainprm": actuator_gainprm,
        "actuator_biasprm": actuator_biasprm,
    })

    return model, in_axes
