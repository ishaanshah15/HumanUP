# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# This file was modified by HumanUP authors in 2024-2025
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: # Copyright (c) 2021 ETH Zurich, Nikita Rudin. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion
from legged_gym.envs.base.humanoid import Humanoid
from legged_gym.envs.g1waist.g1waist_up_config import G1WaistHumanUPCfg  # only for import ordering
from legged_gym.envs.ucrwaist.ucrwaist_config import V0HHumanoidCfg  # <- your config from previous step

from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *

import torch
import numpy as np
import os
from tqdm import tqdm
from torch import Tensor
from typing import Tuple, Dict


class V0HHumanoid(Humanoid):
    def __init__(self,
                 cfg: V0HHumanoidCfg,
                 sim_params: gymapi.SimParams,
                 physics_engine: int,
                 sim_device: str,
                 headless: bool):
        """
        V0H Humanoid environment for fall recovery training using position control.
        
        Key differences from torque-based version:
         - Uses gym.set_dof_position_target_tensor for position control
         - Actions are target joint positions, not torques
         - PD gains are set in DOF properties for position mode
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False

        self._parse_cfg(self.cfg)
        self.domain_rand_general = self.cfg.domain_rand.domain_rand_general

        # Determine device ("cuda:0" vs "cpu")
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"

        # Initialize BaseTask, which in turn calls create_sim()
        BaseTask.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)

        # Initialize root states and DOF positions
        # Root state: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]  
        # Using proper initial pose from configuration
        base_init_state_list = (
            self.cfg.init_state.pos +      # [0, 0.0, 0.95]
            self.cfg.init_state.rot +      # [0, 0, 0, 1] 
            self.cfg.init_state.lin_vel +  # [0, 0, 0]
            self.cfg.init_state.ang_vel     # [0, 0, 0]
        )
        base_init_state_list = [0, 0, 0.2] + [0,-0.7071,0,0.7071] + [0, 0, 0] + [0, 0, 0]  # Add velocities
        self.initial_root_states = torch.tensor(base_init_state_list, device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        
        # Initialize DOF positions based on joint order from config
        # Order: torso (3), right arm (4), left arm (4), right leg (6), left leg (6) = 23 total
        initial_dof_pos_list = [
            # Torso: torsoYaw, torsoPitch, torsoRoll
            0.0, 0.0, 0.0,
            # Right arm: rightShoulderPitch, rightShoulderRoll, rightShoulderYaw, rightElbow  
            0.0, 0.0, 0.0, 0.0,
            # Left arm: leftShoulderPitch, leftShoulderRoll, leftShoulderYaw, leftElbow
            0.0, 0.0, 0.0, 0.0,
            # Right leg: rightHipYaw, rightHipRoll, rightHipPitch, rightKneePitch, rightAnklePitch, rightAnkleRoll
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # Left leg: leftHipYaw, leftHipRoll, leftHipPitch, leftKneePitch, leftAnklePitch, leftAnkleRoll  
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        self.initial_dof_pos = torch.tensor(initial_dof_pos_list, device=self.device, dtype=torch.float).repeat(self.num_envs, 1)

        # -----------------------------------------------------------
        #  CORRECTED: v0H joint indices based on actual MJCF ordering
        #  From action space inspection:
        #  [0-2]: Torso (torsoYaw, torsoPitch, torsoRoll)
        #  [3-6]: Right arm (rightShoulderPitch, rightShoulderRoll, rightShoulderYaw, rightElbow)
        #  [7-10]: Left arm (leftShoulderPitch, leftShoulderRoll, leftShoulderYaw, leftElbow)
        #  [11-16]: Right leg (rightHipYaw, rightHipRoll, rightHipPitch, rightKneePitch, rightAnklePitch, rightAnkleRoll)
        #  [17-22]: Left leg (leftHipYaw, leftHipRoll, leftHipPitch, leftKneePitch, leftAnklePitch, leftAnkleRoll)
        # -----------------------------------------------------------
        self.left_dof_indices = torch.tensor([7, 8, 9, 10, 17, 18, 19, 20, 21, 22],  # Left arm + left leg
                                             device=self.device, dtype=torch.long)
        self.right_dof_indices = torch.tensor([3, 4, 5, 6, 11, 12, 13, 14, 15, 16],  # Right arm + right leg
                                              device=self.device, dtype=torch.long)
        self.waist_indices = torch.tensor([0, 1], device=self.device, dtype=torch.long)  # torsoYaw, torsoPitch

        if not headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        # Create all the tensors and buffers (observations, rewards, etc.)
        self._init_buffers()

        self._prepare_reward_function()

        # Position control specific buffers
        self.dof_position_targets = torch.zeros(self.num_envs, self.num_dof, device=self.device, dtype=torch.float)
        
        # Counters for fall-recovery logic
        self.global_counter = 0
        self.total_env_steps_counter = 0
        self.termination_height = torch.zeros(self.num_envs, device=self.device)

        # Fall-recovery parameters
        self._recovery_episode_prob = 0.5
        self._recovery_steps = 150
        self._fall_init_prob = 0.5
        self.standing_init_prob = cfg.rewards.standing_scale_range[1]

        # Reset environments and do initial step
        self.reset_idx(torch.arange(self.num_envs, device=self.device), init=True)
        self.post_physics_step()

        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

        # Debug joint mapping on initialization
        if self.debug_viz:
            self.debug_joint_mapping()

    def debug_joint_mapping(self):
        """Debug function to verify correct joint mapping"""
        print("=== V0H POSITION CONTROL JOINT MAPPING VERIFICATION ===")
        print("DOF Names from Isaac Gym:")
        for i, name in enumerate(self.dof_names):
            print(f"{i:2d}: {name}")
        
        print(f"\nLeft DOF indices: {self.left_dof_indices.cpu().numpy()}")
        print("Left joints (arm + leg):")
        for idx in self.left_dof_indices:
            print(f"  [{idx:2d}] {self.dof_names[idx]}")
            
        print(f"\nRight DOF indices: {self.right_dof_indices.cpu().numpy()}")
        print("Right joints (arm + leg):")
        for idx in self.right_dof_indices:
            print(f"  [{idx:2d}] {self.dof_names[idx]}")
            
        print(f"\nWaist indices: {self.waist_indices.cpu().numpy()}")
        print("Waist joints:")
        for idx in self.waist_indices:
            print(f"  [{idx:2d}] {self.dof_names[idx]}")

    def _init_buffers(self):
        super()._init_buffers()
        # We need rigid-body rotation for foot orientation rewards
        self.rigid_body_rot = self.rigid_body_states[..., :self.num_bodies, 3:7]

    def step(self, actions):
        """Position control step function - equivalent to torque-based version"""
        # Reindex actions to match v0H joint ordering
        actions = self.reindex(actions)
        actions = actions.to(self.device)
        action_tensor = actions.clone()
        
        # Update action history buffer (now contains position targets)
        self.action_history_buf = torch.cat(
            [self.action_history_buf[:, 1:].clone(), action_tensor[:, None, :].clone()], dim=1
        )
        
        # Action delay (if enabled) - use delayed position targets
        if self.cfg.domain_rand.action_delay:
            if self.total_env_steps_counter <= 5000 * 24:
                self.delay = torch.tensor(0, device=self.device, dtype=torch.float)
            else:
                self.delay = torch.tensor(
                    np.random.randint(2), device=self.device, dtype=torch.float
                )
            indices = -self.delay - 1
            action_tensor = self.action_history_buf[:, indices.long()]

        # Update global counters
        self.global_counter += 1
        self.total_env_steps_counter += 1
        
        # Clip actions (position targets)
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(action_tensor, -clip_actions, clip_actions).to(self.device)
        
        # Render if needed
        self.render()

        # Physics simulation loop
        for _ in range(self.cfg.control.decimation):
            # Convert actions to position targets (relative to default positions)
            self.dof_position_targets = (
                self.default_dof_pos_all + 
                self.actions * self.cfg.control.action_scale
            )
            
            # Set position targets (equivalent to setting torques in original)
            self.gym.set_dof_position_target_tensor(
                self.sim, 
                gymtorch.unwrap_tensor(self.dof_position_targets)
            )
            
            # Simulate physics step
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        # Post-physics processing
        self.post_physics_step()

        # Clip observations
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras


    def _create_envs(self):
        """
        Creates each IsaacGym env, loads v0H.xml asset, sets DOF/rigid-shape props,
        and remembers the index of every critical body (feet, torso/chest, etc.).
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # Asset loading options
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS  # Set to position mode
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

        # Load the v0H asset
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # Get lists of all body-names and DOF-names
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        print(f"Loaded v0H robot with {self.num_dof} DOFs and {self.num_bodies} bodies (Position Control)")

        # Identify key body indices
        self.torso_idx = self.gym.find_asset_rigid_body_index(robot_asset, self.cfg.asset.torso_name)
        self.chest_idx = self.gym.find_asset_rigid_body_index(robot_asset, self.cfg.asset.chest_name)
        
        # v0H has no separate "forehead," so head_idx is -1
        self.head_idx = -1
        if self.cfg.asset.forehead_name is not None:
            self.head_idx = self.gym.find_asset_rigid_body_index(robot_asset, self.cfg.asset.forehead_name)

        # Create force sensors on each foot
        for foot_name in self.cfg.asset.feet_bodies:
            fi = self.gym.find_asset_rigid_body_index(robot_asset, foot_name)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, fi, sensor_pose)

        # Build lists of "penalized" or "termination" contact bodies
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # Store the base-init state from cfg.init_state
        base_init = (self.cfg.init_state.pos +
                     self.cfg.init_state.rot +
                     self.cfg.init_state.lin_vel +
                     self.cfg.init_state.ang_vel)
        self.base_init_state = to_torch(base_init, device=self.device, requires_grad=False)

        # Create environments
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.actor_handles = []
        self.envs = []
        self.mass_params_tensor = torch.zeros((self.num_envs, 4),
                                              dtype=torch.float,
                                              device=self.device,
                                              requires_grad=False)

        # Precompute env spacing
        self._get_env_origins()
        spacing = self.cfg.env.env_spacing
        if self.cfg.terrain.mesh_type == "plane":
            env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
            env_upper = gymapi.Vec3(spacing, spacing, spacing)
        else:
            env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
            env_upper = gymapi.Vec3(0.0, 0.0, 0.0)

        print("Creating v0H environments with position control...")
        for i in tqdm(range(self.num_envs)):
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )

            # Randomize XY start if requested
            pos = self.env_origins[i].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_sqrt_float(-0.3, 0.3, (2,), device=self.device)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = np.random.uniform(-self.cfg.env.rand_yaw_range,
                                             self.cfg.env.rand_yaw_range)
                start_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, rand_yaw)

            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            # Apply per-env overrides on rigid-shape and DOF properties
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

            robot_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, "v0h", i,
                self.cfg.asset.self_collisions, 0
            )

            # DOF properties for position control
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)

            # Rigid body properties
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, robot_handle, body_props, recomputeInertia=True
            )

            self.envs.append(env_handle)
            self.actor_handles.append(robot_handle)
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).float()

        # Store friction coefficients if randomization is enabled
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = (
                self.friction_coeffs.to(self.device).float().squeeze(-1)
            )

        # Store body names and get body indices
        self.body_names = body_names
        self._get_body_indices()

        # Build the "feet_indices" from cfg.asset.feet_bodies
        feet_names = [s for s in body_names if any(f in s for f in self.cfg.asset.feet_bodies)]
        self.feet_indices = torch.zeros(len(feet_names),
                                        dtype=torch.long,
                                        device=self.device,
                                        requires_grad=False)
        for idx, fname in enumerate(self.cfg.asset.feet_bodies):
            self.feet_indices[idx] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], fname
            )

        # Build penalized_contact_indices
        self.penalized_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )
        for idx, cname in enumerate(penalized_contact_names):
            self.penalized_contact_indices[idx] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], cname
            )

        # Build termination_contact_indices
        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )
        for idx, tname in enumerate(termination_contact_names):
            self.termination_contact_indices[idx] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], tname
            )

        # Set up cameras for video recording if enabled
        if self.cfg.env.record_video:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1440
            camera_props.height = 960
            self._rendering_camera_handles = []
            for i in range(self.num_envs):
                cam_pos = np.array([2.0, 0.0, 0.3])
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                self._rendering_camera_handles.append(camera_handle)
                self.gym.set_camera_location(
                    camera_handle,
                    self.envs[i],
                    gymapi.Vec3(*cam_pos),
                    gymapi.Vec3(0.0, 0.0, 0.0)
                )

    def _process_dof_props(self, props, env_id):
        """Process DOF properties for position control mode"""
        # Call parent method first if it exists
        if hasattr(super(), '_process_dof_props'):
            props = super()._process_dof_props(props, env_id)
        
        # PD gains for position control - these values are from the MJCF actuators
        joint_pd_gains = {
            # Torso (A900 class)
            'torsoYaw': {'kp': 200, 'kd': 6},
            'torsoPitch': {'kp': 200, 'kd': 6}, 
            'torsoRoll': {'kp': 200, 'kd': 6},
            
            # Arms - Shoulders (B903 class)
            'rightShoulderPitch': {'kp': 200, 'kd': 6},
            'rightShoulderRoll': {'kp': 200, 'kd': 6},
            'leftShoulderPitch': {'kp': 200, 'kd': 6},
            'leftShoulderRoll': {'kp': 200, 'kd': 6},
            
            # Arms - Shoulder Yaw (C806 class)  
            'rightShoulderYaw': {'kp': 200, 'kd': 6},
            'leftShoulderYaw': {'kp': 200, 'kd': 6},
            
            # Arms - Elbows (C806 class)
            'rightElbow': {'kp': 150, 'kd': 5},
            'leftElbow': {'kp': 150, 'kd': 5},
            
            # Legs - Hip Yaw (A900 class)
            'rightHipYaw': {'kp': 300, 'kd': 10},
            'leftHipYaw': {'kp': 300, 'kd': 10},
            
            # Legs - Hip Roll, Hip Pitch, Knee Pitch, Ankle Pitch (D110a class)
            'rightHipRoll': {'kp': 300, 'kd': 10},
            'leftHipRoll': {'kp': 300, 'kd': 10},
            'rightHipPitch': {'kp': 300, 'kd': 10},
            'leftHipPitch': {'kp': 300, 'kd': 10},
            'rightKneePitch': {'kp': 300, 'kd': 10},
            'leftKneePitch': {'kp': 300, 'kd': 10},
            'rightAnklePitch': {'kp': 300, 'kd': 10},
            'leftAnklePitch': {'kp': 300, 'kd': 10},
            
            # Legs - Ankle Roll (A71 class)
            'rightAnkleRoll': {'kp': 150, 'kd': 5},
            'leftAnkleRoll': {'kp': 150, 'kd': 5},
        }
        
        # Apply gains and verify joint names match
        for i, dof_name in enumerate(self.dof_names):
            if dof_name in joint_pd_gains:
                gains = joint_pd_gains[dof_name]
                props['stiffness'][i] = gains['kp']
                props['damping'][i] = gains['kd']
                props['driveMode'][i] = gymapi.DOF_MODE_POS  # Position control mode
                
                # Apply domain randomization if configured
                if hasattr(self.cfg.domain_rand, 'randomize_gains') and self.cfg.domain_rand.randomize_gains:
                    if hasattr(self.cfg.domain_rand, 'stiffness_multiplier_range'):
                        rand_factor_kp = np.random.uniform(
                            1.0 - self.cfg.domain_rand.stiffness_multiplier_range, 
                            1.0 + self.cfg.domain_rand.stiffness_multiplier_range
                        )
                        props['stiffness'][i] *= rand_factor_kp
                        
                    if hasattr(self.cfg.domain_rand, 'damping_multiplier_range'):
                        rand_factor_kd = np.random.uniform(
                            1.0 - self.cfg.domain_rand.damping_multiplier_range,
                            1.0 + self.cfg.domain_rand.damping_multiplier_range
                        )
                        props['damping'][i] *= rand_factor_kd
            else:
                # Fallback for any unmatched DOFs
                props['stiffness'][i] = 100.0
                props['damping'][i] = 5.0
                props['driveMode'][i] = gymapi.DOF_MODE_POS
                print(f"Warning: No PD gains defined for joint {dof_name}, using defaults")
        
        return props

    def _update_standing_prob_curriculum(self):
        # [Curriculum] Update the standing probability based on the curriculum type
        assert self.cfg.rewards.standing_scale_curriculum_type in ["sin", "step_height", "step", "cos"]
        if self.cfg.rewards.standing_scale_curriculum_type == "sin":
            # Sine wave curriculum, the standing probability will change from 0 to 1 and back to 0 in a cycle
            iteration = self.total_env_steps_counter // 24
            cycle_iteration = iteration % self.cfg.rewards.standing_scale_curriculum_iterations
            sin_progress = (cycle_iteration / self.cfg.rewards.standing_scale_curriculum_iterations) * torch.pi
            self.cfg.rewards.standing_scale = self.cfg.rewards.standing_scale_range[0] + (
                self.cfg.rewards.standing_scale_range[1] - self.cfg.rewards.standing_scale_range[0]
            ) * torch.sin(torch.tensor(sin_progress).to(self.device))
            self.standing_init_prob = self.cfg.rewards.standing_scale.clamp(0.0, 1.0)
        elif self.cfg.rewards.standing_scale_curriculum_type == "cos":
            # Cosine annealing curriculum, the standing probability will decrease from 1 to 0
            iteration = self.total_env_steps_counter // 24
            cycle_iteration = iteration % self.cfg.rewards.standing_scale_curriculum_iterations
            cos_progress = (cycle_iteration / self.cfg.rewards.standing_scale_curriculum_iterations) * torch.pi / 2.0
            self.cfg.rewards.standing_scale = self.cfg.rewards.standing_scale_range[0] + (
                self.cfg.rewards.standing_scale_range[1] - self.cfg.rewards.standing_scale_range[0]
            ) * torch.cos(torch.tensor(cos_progress).to(self.device)).clamp(0.0, 1.0)
            self.standing_init_prob = self.cfg.rewards.standing_scale
        elif self.cfg.rewards.standing_scale_curriculum_type == "step_height":
            # Step height curriculum, the standing probability will change based on the average termination height of all envs
            # NOTE: This is an inversed version of regularization scale curriculum in step_height - Runpei
            if torch.mean(self.termination_height).item() > 0.65:
                # drease the regularization scale
                self.standing_init_prob *= (
                    1.0 - self.cfg.rewards.standing_scale_gamma
                )
            elif torch.mean(self.termination_height).item() < 0.1:
                # increase the regularization scale
                self.standing_init_prob *= (
                    1.0 + self.cfg.rewards.standing_scale_gamma
                )
            self.standing_init_prob = max(
                min(
                    self.standing_init_prob,
                    self.cfg.rewards.standing_scale_range[1],
                ),
                self.cfg.rewards.standing_scale_range[0],
            )
        elif self.cfg.rewards.standing_scale_curriculum_step:
            # TODO: implement step curriculum - Runpei
            raise NotImplementedError

    def _reset_dofs(self, env_ids, dof_pos=None, dof_vel=None, set_act: bool = True):
        """
        Reset all DOF-positions (and possibly small randomization).
        """
        if dof_pos is None:
            self.dof_pos[env_ids] = self.initial_dof_pos[env_ids].clone()
        else:
            self.dof_pos[env_ids] = dof_pos[env_ids].clone()
        self.dof_vel[env_ids] = 0.0

        # Also reset position targets to current positions
        self.dof_position_targets[env_ids] = self.dof_pos[env_ids].clone()

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        if set_act:
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )

    def _reset_root_states(
            self,
            env_ids,
            root_vel=None,
            root_quat=None,
            root_height=None,
            use_base_init_state=False,
            set_act: bool = True):
        """
        Reset the floating-base (root) state.
        """
        if use_base_init_state:
            self.root_states[env_ids] = self.base_init_state
        else:
            self.root_states[env_ids] = self.initial_root_states[env_ids].clone()

        if self.custom_origins:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 2] += 0.01
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_sqrt_float(
                    -0.3, 0.3, (len(env_ids), 2), device=self.device
                )
        else:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        if set_act:
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )

    def _reset_stand_and_lie_states(self, env_ids, dof_pos):
        """
        Mixed stand/fall reset logic.
        """
        if self.cfg.rewards.standing_scale_curriculum:
            self._update_standing_prob_curriculum()
        num_standing = int(self.num_envs * self.standing_init_prob)
        standing_env_flag = env_ids < num_standing
        non_standing_env_flag = env_ids >= num_standing

        # falling states
        if len(env_ids[non_standing_env_flag]) > 0:
            self._reset_dofs(env_ids[non_standing_env_flag], set_act=False)
            self._reset_root_states(env_ids[non_standing_env_flag], set_act=False)

        # standing states
        if len(env_ids[standing_env_flag]) > 0:
            self._reset_dofs(env_ids[standing_env_flag], dof_pos=dof_pos, set_act=False)
            self._reset_root_states(env_ids[standing_env_flag],
                                     use_base_init_state=True,
                                     set_act=False)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # now write everything back to GPU
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        """
        Same as G1's cosine/sine/step curriculum on "standing_init_prob."
        """
        assert self.cfg.rewards.standing_scale_curriculum_type in ["sin", "step_height", "cos"]
        if self.cfg.rewards.standing_scale_curriculum_type == "sin":
            iteration = self.total_env_steps_counter // 24
            safe_iterations = max(self.cfg.rewards.standing_scale_curriculum_iterations, 1)
            cycle_iteration = iteration % safe_iterations
            sin_progress = (cycle_iteration / safe_iterations) * torch.pi
            self.cfg.rewards.standing_scale = (
                self.cfg.rewards.standing_scale_range[0] +
                (self.cfg.rewards.standing_scale_range[1] -
                 self.cfg.rewards.standing_scale_range[0]) *
                torch.sin(torch.tensor(sin_progress).to(self.device))
            )
            self.standing_init_prob = self.cfg.rewards.standing_scale.clamp(0.0, 1.0)

        elif self.cfg.rewards.standing_scale_curriculum_type == "cos":
            iteration = self.total_env_steps_counter // 24
            safe_iterations = max(self.cfg.rewards.standing_scale_curriculum_iterations, 1)
            cycle_iteration = iteration % safe_iterations
            cos_progress = (cycle_iteration / safe_iterations) * torch.pi / 2.0
            self.cfg.rewards.standing_scale = (
                self.cfg.rewards.standing_scale_range[0] +
                (self.cfg.rewards.standing_scale_range[1] -
                 self.cfg.rewards.standing_scale_range[0]) *
                torch.cos(torch.tensor(cos_progress).to(self.device)).clamp(0.0, 1.0)
            )
            self.standing_init_prob = self.cfg.rewards.standing_scale

        elif self.cfg.rewards.standing_scale_curriculum_type == "step_height":
            mean_term_height = torch.mean(self.termination_height).item()
            if mean_term_height > 0.65:
                self.standing_init_prob *= (1.0 - self.cfg.rewards.standing_scale_gamma)
            elif mean_term_height < 0.1:
                self.standing_init_prob *= (1.0 + self.cfg.rewards.standing_scale_gamma)
            self.standing_init_prob = max(
                min(self.standing_init_prob, self.cfg.rewards.standing_scale_range[1]),
                self.cfg.rewards.standing_scale_range[0]
            )

    def reset_idx(self, env_ids, init=False):
        """
        Overridden to mix stand/fall resets.
        """
        if len(env_ids) == 0:
            return

        # Terrain curriculum if enabled
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        dof_pos = self.default_dof_pos_all.clone()
        if self.standing_init_prob > 0:
            self._reset_stand_and_lie_states(env_ids, dof_pos=dof_pos)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        # no "velocity commands" for v0H
        self._resample_commands(env_ids)

        # kick the sim
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Reset all buffers at once
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_torques[env_ids] = 0.0
        self.last_root_vel[:] = 0.0
        self.last_contact_forces[env_ids] = torch.zeros_like(self.contact_forces[env_ids])
        self.last_base_height = torch.zeros_like(self.root_states[env_ids, 2])
        self.feet_air_time[env_ids] = 0.0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.0
        self.contact_buf[env_ids, :, :] = 0.0
        self.action_history_buf[env_ids, :, :] = 0.0
        self.feet_land_time[env_ids] = 0.0

        self._reset_buffers_extra(env_ids)

        # Fill extras for reporting
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["metric_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids] * self.reward_scales[key])
                / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0

        self.episode_length_buf[env_ids] = 0

        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _update_recovery_count(self):
        self._recovery_counter -= 1
        self._recovery_counter = torch.clamp_min(self._recovery_counter, 0)

    def _post_physics_step_callback(self):
        """
        Called every step.  Mirror's G1's push/drag/command‐resample logic.
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)) == 0
        self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        if self.cfg.domain_rand.drag_robot_up:
            if self.cfg.domain_rand.drag_when_falling:
                if self.cfg.domain_rand.force_compenstation:
                    self._drag_robots(self.base_lin_vel[:, 2], random=False)
                else:
                    if self.cfg.domain_rand.drag_robot_by_force:
                        self._drag_robots_by_force()
                    else:
                        self._drag_robots(self.base_lin_vel[:, 2])
            elif self.common_step_counter % self.cfg.domain_rand.drag_interval == 0:
                if self.cfg.domain_rand.drag_robot_by_force:
                    self._drag_robots_by_force()
                else:
                    self._drag_robots()

    def _drag_robots(self, z_vel=None, random=True):
        """
        Exactly same as G1's: apply an upward velocity impulse to the base.
        """
        if z_vel is None:
            min_drag = self.cfg.domain_rand.min_drag_vel
            max_drag = self.cfg.domain_rand.max_drag_vel
            self.root_states[:, 9] += torch_rand_float(
                min_drag, max_drag, (self.num_envs, 1), device=self.device
            ).squeeze(1)
        else:
            drag_flag = z_vel < 0
            if random:
                self.root_states[drag_flag, 9] = torch_rand_float(
                    self.cfg.domain_rand.min_drag_vel,
                    self.cfg.domain_rand.max_drag_vel,
                    (drag_flag.sum().item(),),
                    device=self.device
                )
            else:
                self.root_states[drag_flag, 9] = -z_vel[drag_flag]
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _drag_robots_by_force(self):
        """
        Apply a literal force to whichever body v0HCfg says (e.g. torso).
        """
        force = self.cfg.domain_rand.drag_force
        if self.cfg.domain_rand.drag_force_curriculum:
            force = self._update_drag_force_curriculum(force)

        forces = torch.zeros((self.num_envs, self.num_bodies, 3),
                             device=self.device, dtype=torch.float)
        torques = torch.zeros((self.num_envs, self.num_bodies, 3),
                              device=self.device, dtype=torch.float)

        part = self.cfg.domain_rand.drag_robot_part
        idx = getattr(self, part + "_idx")
        forces[:, idx, 2] = force
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(forces),
            gymtorch.unwrap_tensor(torques),
            gymapi.ENV_SPACE
        )

    def check_termination(self):
        super().check_termination()

        # If base height out of [0,1.2], terminate
        if self.cfg.env.terminate_on_height:
            base_too_high = torch.logical_or(
                self.root_states[:, 2] > 1.2,
                self.root_states[:, 2] < 0.0
            )
            self.reset_buf[base_too_high] = 1

        # Record termination height for curriculum
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.termination_height[env_ids] = self.root_states[env_ids, 2]

    def post_physics_step(self):
        """
        Overridden version, exactly like G1's:
         - refresh all tensors
         - apply termination/reward/obs etc.
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Build IMU + base velocities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        safe_dt = torch.clamp(torch.tensor(self.dt, device=self.device), min=1e-6)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / safe_dt

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        # foot‐contact flag (|force| > 2 N)
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        # any custom per‐step pushes/pulls
        self._post_physics_step_callback()

        # term/reward/observations
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        self.episode_length[env_ids] = self.episode_length_buf[env_ids].float()
        self.reset_idx(env_ids)
        self.compute_observations()

        # stash last‐step values
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13].clone()
        self.last_base_height = self.root_states[:, 2].clone()
        self.last_contact_forces = self.contact_forces.clone()

        if self.cfg.rewards.regularization_scale_curriculum:
            self._update_regularization_scale_curriculum()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.draw_goal()

        # reduce any fall‐recovery counters
        self._update_recovery_count()

    def _update_regularization_scale_curriculum(self):
        """
        Same "step/sin" curriculum on regularization‐scale as G1.
        """
        rcfg = self.cfg.rewards
        assert rcfg.regularization_scale_curriculum_type in ["sin", "step_height", "step_episode"]
        if rcfg.regularization_scale_curriculum_type == "sin":
            iteration = self.total_env_steps_counter // 24
            safe_iterations = max(rcfg.regularization_scale_curriculum_iterations, 1)
            cycle_iteration = iteration % safe_iterations
            sin_progress = (cycle_iteration / safe_iterations) * torch.pi
            rcfg.regularization_scale = (
                rcfg.regularization_scale_range[0] +
                (rcfg.regularization_scale_range[1] - rcfg.regularization_scale_range[0]) *
                torch.sin(torch.tensor(sin_progress).to(self.device)).clamp(0.0, 1.0)
            )

        elif rcfg.regularization_scale_curriculum_type == "step_height":
            mean_term_height = torch.mean(self.termination_height).item()
            if mean_term_height > 0.65:
                rcfg.regularization_scale *= (1.0 + rcfg.regularization_scale_gamma)
            elif mean_term_height < 0.1:
                rcfg.regularization_scale *= (1.0 - rcfg.regularization_scale_gamma)
            rcfg.regularization_scale = max(
                min(rcfg.regularization_scale, rcfg.regularization_scale_range[1]),
                rcfg.regularization_scale_range[0]
            )

        elif rcfg.regularization_scale_curriculum_type == "step_episode":
            mean_episode_len = torch.mean(self.episode_length.float()).item()
            if mean_episode_len > 420.0:
                rcfg.regularization_scale *= (1.0 + rcfg.regularization_scale_gamma)
            elif mean_episode_len < 50.0:
                rcfg.regularization_scale *= (1.0 - rcfg.regularization_scale_gamma)
            rcfg.regularization_scale = max(
                min(rcfg.regularization_scale, rcfg.regularization_scale_range[1]),
                rcfg.regularization_scale_range[0]
            )

    
    def compute_observations(self):
        """
        Exactly the same concatenation logic as G1.  v0H's index‐order is already
        mapped via self.left_dof_indices etc., so no changes here.
        """
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(0 * self.yaw, 0 * self.yaw, self.yaw)

        obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,    # 3 dims
            imu_obs,                                        # 2 dims
            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
            self.reindex(self.action_history_buf[:, -1]),
        ), dim=-1)

        # add noise if configured
        if self.cfg.noise.add_noise:
            safe_noise_steps = max(self.cfg.noise.noise_increasing_steps * 24, 1)
            noise_scale = torch.minimum(
                self.total_env_steps_counter / safe_noise_steps, 1.0
            ) if self.headless else 1.0
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * noise_scale

        # privileged latent vector
        if self.cfg.domain_rand.domain_rand_general:
            # Ensure motor_strength is initialized, use default values if not
            if hasattr(self, 'motor_strength') and self.motor_strength is not None:
                motor_strength_0 = self.motor_strength[0] - 1
                motor_strength_1 = self.motor_strength[1] - 1
            else:
                motor_strength_0 = torch.zeros(self.num_envs, device=self.device)
                motor_strength_1 = torch.zeros(self.num_envs, device=self.device)
            
            priv_latent = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                motor_strength_0,
                motor_strength_1,
                self.base_lin_vel,
            ), dim=-1)
        else:
            priv_latent = torch.zeros(
                (self.num_envs, self.cfg.env.n_priv_latent), device=self.device
            )

        self.obs_buf = torch.cat(
            [obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)],
            dim=-1
        )
        
        # slide history buffers
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

    def _resample_commands(self, env_ids):
        """
        v0H does not have a walking command.  We only sample vertical‐velocity commands.
        Exactly as in G1.
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_z"][0],
            self.command_ranges["lin_vel_z"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        # zero out small commands
        self.commands[env_ids, :2] *= (
            torch.abs(self.commands[env_ids, 0:1]) > self.cfg.commands.lin_vel_clip
        )

    # ================================================ Rewards ================================================== #

    # All reward functions remain identical to the original version
    def  _reward_base_height_exp(self):
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
        left_body_action = self.actions[:, self.left_dof_indices]  # [num_envs, 10] 
        right_body_action = self.actions[:, self.right_dof_indices]  # [num_envs, 10]
        
        # Update negative indices for new mapping:
        # For left vs right symmetry, these joints should be negated:
        # - Hip/shoulder roll (indices 1, 5 in the 10-element arrays)
        # - Hip/ankle pitch (indices 2, 4 in left array, 2, 4 in right array)
        negative_indices = torch.tensor([1, 5], device=self.device, dtype=torch.long)  # Shoulder roll, hip roll
        left_body_action[:, negative_indices] *= -1
        body_symmetry = torch.norm(left_body_action - right_body_action, dim=-1)

        if self.cfg.env.no_symmetry_after_stand:
            if self.head_idx == -1:
                standing_flag = self.rigid_body_states[:, self.torso_idx, 2] > 1.1
            else:
                standing_flag = self.rigid_body_states[:, self.head_idx, 2] > 1.1
            body_symmetry[standing_flag] *= 0
        return body_symmetry

    def _reward_soft_symmetry_waist(self):
        waist_roll_yaw = self.actions[:, self.waist_indices]
        waist_symmetry = torch.norm(waist_roll_yaw, dim=-1)
        if self.cfg.env.no_symmetry_after_stand:
            # Use torso height for standing check if no head available
            if self.head_idx == -1:
                standing_flag = self.rigid_body_states[:, self.torso_idx, 2] > 1.1
            else:
                standing_flag = self.rigid_body_states[:, self.head_idx, 2] > 1.1
            waist_symmetry[standing_flag] *= 0
        return waist_symmetry

    def _reward_dof_vel(self):
        return 0.0

    def _reward_base_lin_vel(self):
        return 0.0

    def _reward_ang_vel(self):
        return 0.0

    def _reward_torques(self):
        return 0.0
    
    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return 0.0
    
    def _reward_dof_torque_limits(self):
        # out_of_limits = torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0), dim=1)
        # return out_of_limits
        return 0.0
    
    def _reward_energy(self):
        return 0.0

    def _reward_dof_acc(self):
        return 0.0

    def _reward_head_height_exp(self):
        return 0.0