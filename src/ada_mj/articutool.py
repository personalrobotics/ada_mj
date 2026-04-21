# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Articutool 2-DOF entity — placeholder for Slice 3."""

from __future__ import annotations

import mujoco
import numpy as np

from ada_mj.config import ArticutoolConfig


class Articutool:
    """2-DOF motorized tool: tilt + roll, with F/T sensor and IMU.

    Implements the entity protocol for SimContext registration.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: ArticutoolConfig):
        self._model = model
        self._data = data
        self._config = config

        # Resolve joint/actuator/sensor IDs
        self._tilt_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, config.tilt_joint)
        self._roll_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, config.roll_joint)
        self._tilt_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, config.tilt_actuator)
        self._roll_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, config.roll_actuator)

        self._tilt_qadr = model.jnt_qposadr[self._tilt_jnt]
        self._roll_qadr = model.jnt_qposadr[self._roll_jnt]
        self._tilt_vadr = model.jnt_dofadr[self._tilt_jnt]
        self._roll_vadr = model.jnt_dofadr[self._roll_jnt]

        # Sensor IDs
        self._ft_force_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, config.ft_force_sensor)
        self._ft_torque_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, config.ft_torque_sensor)
        self._imu_accel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, config.imu_accel_sensor)
        self._imu_gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, config.imu_gyro_sensor)

        # Weld constraint ID
        self._weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, config.weld_name)

    # -- Entity protocol (for SimContext) -------------------------------------

    @property
    def name(self) -> str:
        return "articutool"

    @property
    def joint_qpos_indices(self) -> list[int]:
        return [self._tilt_qadr, self._roll_qadr]

    @property
    def joint_qvel_indices(self) -> list[int]:
        return [self._tilt_vadr, self._roll_vadr]

    @property
    def actuator_ids(self) -> list[int]:
        return [self._tilt_act, self._roll_act]

    # -- Tilt/Roll ------------------------------------------------------------

    def get_tilt(self) -> float:
        return float(self._data.qpos[self._tilt_qadr])

    def get_roll(self) -> float:
        return float(self._data.qpos[self._roll_qadr])

    def set_tilt(self, angle: float) -> None:
        self._data.qpos[self._tilt_qadr] = angle

    def set_roll(self, angle: float) -> None:
        self._data.qpos[self._roll_qadr] = angle

    def set_tilt_roll(self, tilt: float, roll: float) -> None:
        self.set_tilt(tilt)
        self.set_roll(roll)

    # -- Sensors --------------------------------------------------------------

    def get_ft_wrench(self) -> np.ndarray:
        """[fx,fy,fz,tx,ty,tz] from the Hex21 F/T sensor."""
        m = self._model
        force_adr = m.sensor_adr[self._ft_force_id]
        torque_adr = m.sensor_adr[self._ft_torque_id]
        force = self._data.sensordata[force_adr:force_adr + 3].copy()
        torque = self._data.sensordata[torque_adr:torque_adr + 3].copy()
        return np.concatenate([force, torque])

    def get_imu(self) -> tuple[np.ndarray, np.ndarray]:
        """(accel[3], gyro[3]) from the ICM-20948 IMU."""
        m = self._model
        accel_adr = m.sensor_adr[self._imu_accel_id]
        gyro_adr = m.sensor_adr[self._imu_gyro_id]
        accel = self._data.sensordata[accel_adr:accel_adr + 3].copy()
        gyro = self._data.sensordata[gyro_adr:gyro_adr + 3].copy()
        return accel, gyro

    # -- Weld management ------------------------------------------------------

    def is_attached(self) -> bool:
        return bool(self._data.eq_active[self._weld_id])

    def detach(self) -> None:
        self._data.eq_active[self._weld_id] = 0

    def attach(self) -> None:
        self._data.eq_active[self._weld_id] = 1
