# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# This file was modified by HumanUP authors in 2024-2025
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: # Copyright (c) 2021 ETH Zurich, Nikita Rudin. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024-2025 RoboVision Lab, UIUC. All rights reserved.

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from tqdm import tqdm
from warnings import WarningMessage
import numpy as np
import os
import cv2

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion
from legged_gym.gym_utils.terrain import Terrain
from legged_gym.gym_utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.gym_utils.helpers import class_to_dict
from legged_gym.envs.base.humanoid import Humanoid
from legged_gym.envs.base.humanoid_config import HumanoidCfg, HumanoidCfgPPO
from legged_gym.envs.v0h.v0h_config import V0HHumanoidCfg


class V0HHumanoid(Humanoid):
    def __init__(self, cfg: V0HHumanoidCfg, sim_params, physics_engine, sim_device, headless):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.domain_rand_general = self.cfg.domain_rand.domain_rand_general

        # Pre init for motion loading
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"

        BaseTask.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)

        # V0H-specific initial states - adapted from MuJoCo configuration
        self.initial_root_states = torch.tensor([
            0.0, 0.0, 0.95,  # position: x, y, z (pelvis at 0.95m height)
            0.0, 0.0, 0.0, 1.0,  # orientation: quat (x, y, z, w)
            0.0, 0.0, 0.0,  # linear velocity
            0.0, 0.0, 0.0   # angular velocity
        ]).to(sim_device).repeat(self.num_envs, 1)

        # V0H joint order matches the MuJoCo actuator order:
        # [torsoYaw, torsoPitch, torsoRoll, 
        #  rightShoulderPitch, rightShoulderRoll, rightShoulderYaw, rightElbow,
        #  leftShoulderPitch, leftShoulderRoll, leftShoulderYaw, leftElbow,
        #  rightHipYaw, rightHipRoll, rightHipPitch, rightKneePitch, rightAnklePitch, rightAnkleRoll,
        #  leftHipYaw, leftHipRoll, leftHipPitch, leftKneePitch, leftAnklePitch, leftAnkleRoll]
        self.initial_dof_pos = torch.tensor([
            0.0, 0.0, 0.0,  # torso joints
            0.0, 0.0, 0.0, 0.0,  # right arm
            0.0, 0.0, 0.0, 0.0,  # left arm
            0.0, 0.0, -0.2, 0.4, -0.2, 0.0,  # right leg (slight knee bend)
            0.0, 0.0, -0.2, 0.4, -0.2, 0.0   # left leg (slight knee bend)
        ]).to(sim_device).repeat(self.num_envs, 1)
        
        # Define joint indices for V0H (based on the actuator order)
        self.left_dof_indices = torch.tensor([
            7, 8, 9, 10,  # left arm: leftShoulderPitch, leftShoulderRoll, leftShoulderYaw, leftElbow
            17, 18, 19, 20, 21, 22  # left leg: leftHipYaw, leftHipRoll, leftHipPitch, leftKneePitch, leftAnklePitch, leftAnkleRoll
        ], device=self.device, dtype=torch.long)
        
        self.right_dof_indices = torch.tensor([
            3, 4, 5, 6,  # right arm: rightShoulderPitch, rightShoulderRoll, rightShoulderYaw, rightElbow
            11, 12, 13, 14, 15, 16  # right leg: rightHipYaw, rightHipRoll, rightHipPitch, rightKneePitch, rightAnklePitch, rightAnkleRoll
        ], device=self.device, dtype=torch.long)
        
        self.torso_indices = torch.tensor([0, 1, 2], device=self.device, dtype=torch.long)  # torsoYaw, torsoPitch, torsoRoll
        
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()

        self.global_counter = 0
        self.total_env_steps_counter = 0

        self.termination_height = torch.zeros(self.num_envs, device=self.device)

        # fall down states
        self._recovery_episode_prob = 0.5
        self._recovery_steps = 150
        self._fall_init_prob = 0.5

        self.standing_init_prob = cfg.rewards.standing_scale_range[1]
        self.reset_idx(torch.arange(self.num_envs, device=self.device), init=True)
        self.post_physics_step()

        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

    def _init_buffers(self):
        super()._init_buffers()
        self.rigid_body_rot = self.rigid_body_states[..., :self.num_bodies, 3:7]

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
            2.1 creates the environment,
            2.2 calls DOF and Rigid shape properties callbacks,
            2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        
        # Find body indices for V0H
        self.torso_idx = self.gym.find_asset_rigid_body_index(robot_asset, self.cfg.asset.torso_name)
        self.chest_idx = self.gym.find_asset_rigid_body_index(robot_asset, self.cfg.asset.chest_name)
        # V0H doesn't have a separate head, so we'll use torso for head-related operations
        self.head_idx = self.torso_idx

        # Create force sensors for feet
        for s in self.cfg.asset.feet_bodies:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []

        # record the initial standing default state
        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        
        self._get_env_origins()
        spacing = self.cfg.env.env_spacing
        if self.cfg.terrain.mesh_type == "plane":
            env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
            env_upper = gymapi.Vec3(spacing, spacing, spacing)
        else:
            env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
            env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )

        print("Creating V0H environments...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(
                    0.0, 0.0, self.cfg.env.rand_yaw_range * np.random.uniform(-1, 1)
                )
                start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            robot_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, "v0h", i, self.cfg.asset.self_collisions, 0
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, robot_handle, body_props, recomputeInertia=True
            )
            self.envs.append(env_handle)
            self.actor_handles.append(robot_handle)

            self.mass_params_tensor[i, :] = (
                torch.from_numpy(mass_params).to(self.device).to(torch.float)
            )
        
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = (
                self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)
            )

        self.body_names = body_names
        self._get_body_indices()

        # Find feet indices
        feet_names = self.cfg.asset.feet_bodies
        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        self.penalized_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

        if self.cfg.env.record_video:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 720 * 2
            camera_props.height = 480 * 2
            self._rendering_camera_handles = []
            for i in range(self.num_envs):
                cam_pos = np.array([2, 0, 0.3])
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                self._rendering_camera_handles.append(camera_handle)
                self.gym.set_camera_location(
                    camera_handle, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*0 * cam_pos)
                )

    def _reset_dofs(self, env_ids, dof_pos=None, dof_vel=None, set_act=True):
        if dof_pos is None:
            self.dof_pos[env_ids] = self.initial_dof_pos[env_ids].clone()
        else:
            self.dof_pos[env_ids] = dof_pos[env_ids].clone()
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        if set_act:
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )

    def _reset_root_states(self, env_ids, root_vel=None, root_quat=None, root_height=None, use_base_init_state=False, set_act=True):
        """Resets ROOT states position and velocities of selected environments"""
        if use_base_init_state:
            self.root_states[env_ids] = self.base_init_state
        else:
            self.root_states[env_ids] = self.initial_root_states[env_ids].clone()
        
        if self.custom_origins:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 2] += 0.01
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(
                    -0.3, 0.3, (len(env_ids), 2), device=self.device
                )
        else:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        if set_act is True:
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )

    # Inherit most methods from the G1 implementation, but adapt specific reward functions
    

    

    

   
    

    def compute_observations(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(0 * self.yaw, 0 * self.yaw, self.yaw)
        
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3 dims
                imu_obs,  # 2 dims
                self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                self.reindex(self.action_history_buf[:, -1]),
            ),
            dim=-1,
        )
        
        if self.cfg.noise.add_noise and self.headless:
            obs_buf += (
                (2 * torch.rand_like(obs_buf) - 1)
                * self.noise_scale_vec
                * min(
                    self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24), 1.0
                )
            )
        elif self.cfg.noise.add_noise and not self.headless:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec
        else:
            obs_buf += 0.0

        if self.cfg.domain_rand.domain_rand_general:
            priv_latent = torch.cat(
                (
                    self.mass_params_tensor,
                    self.friction_coeffs_tensor,
                    self.motor_strength[0] - 1,
                    self.motor_strength[1] - 1,
                    self.base_lin_vel,
                ),
                dim=-1,
            )
        else:
            priv_latent = torch.zeros(
                (self.num_envs, self.cfg.env.n_priv_latent), device=self.device
            )

        self.obs_buf = torch.cat(
            [obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1
        )

        if self.cfg.env.history_len > 0:
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None],
                torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
                torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1),
            )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([self.contact_buf[:, 1:], self.contact_filt.float().unsqueeze(1)], dim=1),
        )

    # Inherit the remaining methods from parent classes
    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments"""
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_z"][0],
            self.command_ranges["lin_vel_z"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        # set small commands to zero
        self.commands[env_ids, :2] *= (
            torch.abs(self.commands[env_ids, 0:1]) > self.cfg.commands.lin_vel_clip
        )
    
    # ================================================ Rewards ================================================== #

    # All reward functions remain identical to the original version
    def _reward_base_height_exp(self):
        z_rwd = torch.clamp(
            self.root_states[:, 2], min=0.0, max=self.cfg.rewards.base_height_target
        )
        return torch.exp(z_rwd) - 1.0

    def _reward_delta_base_height(self):
        base_height = self.root_states[:, 2]
        delta_height = base_height - self.last_base_height
        rise_up = delta_height > 0
        rew = torch.ones_like(base_height) 
        rew[~rise_up] = 0.0
        return rew

    def _reward_head_height_exp(self):
        # For V0H, use torso height since there's no separate head
        z_rwd = torch.clamp(
            self.rigid_body_states[:, self.torso_idx, 2], min=0.0, max=self.cfg.rewards.head_height_target
        )
        return torch.exp(z_rwd) - 1.0

    def _reward_feet_contact_forces_increase(self):
        feet_contact_forces = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)
        last_feet_contact_forces = torch.norm(self.last_contact_forces[:, self.feet_indices, 2], dim=-1)
        delta_contact_forces = feet_contact_forces - last_feet_contact_forces
        increase = delta_contact_forces > 0
        rew = torch.ones_like(feet_contact_forces) * 1.0
        rew[~increase] = 0.0
        return rew

    def _reward_stand_on_feet(self):
        # reward for standing on both feet
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.0
        stand_on_both = torch.sum(contact, dim=1) == 2
        feet_on_ground = self.rigid_body_states[:, self.feet_indices, 2] < 0.1
        feet_on_ground_both = torch.sum(feet_on_ground, dim=1) == 2
        stand_on_both &= feet_on_ground_both
        rew = torch.ones_like(stand_on_both) * 1.0
        rew[~stand_on_both] = 0.0
        return rew
    
    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=-1)
    
    def _reward_body_up_exp(self):
        z_axis = self.projected_gravity[:, 2]  # + down/ - up
        reward = torch.exp(-z_axis)
        return reward

    def _reward_feet_height(self):
        feet_height = torch.mean(self.rigid_body_states[:, self.feet_indices, 2], dim=-1)
        return torch.exp(-10 * feet_height)

    def _reward_feet_orientation(self):
        left_quat = self.rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1) **0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_soft_symmetry_body(self):
        left_body_action = self.actions[:, self.left_dof_indices]
        right_body_action = self.actions[:, self.right_dof_indices]
        
        # For V0H, we need to flip signs for certain joints to enforce symmetry
        # This depends on the joint orientations in the V0H model
        negative_indices = torch.tensor([1, 2, 5, 7, 8], device=self.device, dtype=torch.int64)
        left_body_action[:, negative_indices] *= -1
        body_symmetry = torch.norm(left_body_action - right_body_action, dim=-1)

        if self.cfg.env.no_symmetry_after_stand:
            standing_flag = self.rigid_body_states[:, self.torso_idx, 2] > 1.1
            body_symmetry[standing_flag] *= 0
        return body_symmetry

    def _reward_soft_symmetry_waist(self):
        # V0H has 3 torso joints: torsoYaw, torsoPitch, torsoRoll
        torso_actions = self.actions[:, self.torso_indices]
        torso_symmetry = torch.norm(torso_actions, dim=-1)
        
        if self.cfg.env.no_symmetry_after_stand:
            standing_flag = self.rigid_body_states[:, self.torso_idx, 2] > 1.1
            torso_symmetry[standing_flag] *= 0
        return torso_symmetry

    

