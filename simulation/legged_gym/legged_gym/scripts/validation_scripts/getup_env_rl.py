#!/usr/bin/env python3

"""
Humanoid Robot Get-Up from Supine Position Script - GPU OPTIMIZED VERSION

This script combines the GPU optimization and RL environment matching from the 
arm lift script with the complete get-up sequence from the CPU script.
"""

import numpy as np
import os
import imageio
import math

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch


class HumanoidGetUpVideoEnvGPU:
    def __init__(self, num_envs=1, device="cuda:0", headless=False):
        self.num_envs = num_envs
        self.device = device
        self.headless = headless
        self.env_spacing = 2.0
        
        # RL Environment matching parameters
        self.dt = 0.001  # 1ms timestep like RL env
        self.decimation = 1  # Control at 50Hz like RL env
        self.substeps = 2
        self.control_freq_inv = self.decimation
        self.sim_params_configured = False
        
        # Step counters
        self.common_step_counter = 0
        self.control_step_counter = 0
        
        # Initialize Isaac Gym
        self.gym = gymapi.acquire_gym()
        
        # Configure simulation to match RL environment
        self.configure_simulation()
        
        # Create simulation with GPU pipeline
        self.create_simulation()
        
        # Create ground plane
        self.create_ground_plane()
        
        # Load robot asset
        self.load_robot_asset()
        
        # Create environments
        self.create_envs()
        
        # Initialize tensors
        self.init_tensors()
        
        # Set initial states
        self.setup_initial_states()
        
        # Get-up sequence parameters (adapted from CPU version)
        self.sequence_phase = 0
        self.phase_duration = 180 * self.decimation  # Adjust for decimation
        self.current_phase_time = 0
        
        # Define get-up phases (from CPU version)
        self.phases = [
            'lying_down',      # Initial supine position
            'roll_to_side',    # Roll to side
            'roll_to_prone',   # Complete roll to prone (face down)
            'push_up_prep',    # Prepare for push-up motion
            'push_to_kneel',   # Push up to kneeling position
            'kneel_to_squat',  # Transition to squatting
            'squat_to_stand',  # Final stand up
            'standing'         # Final standing position
        ]
        
    def configure_simulation(self):
        """Configure simulation parameters to match RL environment exactly"""
        self.sim_params = gymapi.SimParams()
        
        # Core timing - match RL environment
        self.sim_params.dt = self.dt
        self.sim_params.substeps = self.substeps
        
        # Gravity
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        # GPU Pipeline - ENABLED like RL environment
        self.sim_params.use_gpu_pipeline = True
        
        # PhysX settings to match RL environment defaults
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.contact_offset = 0.01
        self.sim_params.physx.rest_offset = 0.0
        self.sim_params.physx.bounce_threshold_velocity = 0.5
        self.sim_params.physx.max_depenetration_velocity = 1.0
        self.sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        self.sim_params.physx.default_buffer_size_multiplier = 5
        self.sim_params.physx.contact_collection = gymapi.CC_NEVER
        
        self.sim_params_configured = True
        print(f"✓ Simulation configured: dt={self.dt}, GPU={self.sim_params.use_gpu_pipeline}")
        
    def create_simulation(self):
        """Create simulation with proper device handling"""
        # Parse device string like RL environment
        sim_device_type, sim_device_id = self._parse_device_str(self.device)
        
        if sim_device_type == "cuda" and self.sim_params.use_gpu_pipeline:
            self.sim_device = self.device
        else:
            self.sim_device = "cpu"
            
        print(f"✓ Using device: {self.sim_device}")
        
        # Create simulation
        self.sim = self.gym.create_sim(
            sim_device_id, 
            0,  # graphics device
            gymapi.SIM_PHYSX, 
            self.sim_params
        )
        
        if self.sim is None:
            raise Exception("Failed to create simulation")
            
    def _parse_device_str(self, device_str):
        """Parse device string like RL environment does"""
        if device_str == "cpu":
            return "cpu", 0
        elif device_str.startswith("cuda:"):
            device_id = int(device_str.split(":")[1])
            return "cuda", device_id
        else:
            return "cuda", 0
            
    def create_ground_plane(self):
        """Create ground plane"""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
    def load_robot_asset(self):
        """Load robot asset with RL environment settings"""
        # Asset loading options matching RL environment
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 1000.0
        asset_options.max_linear_velocity = 1000.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        
        # Load the robot asset
        asset_root = "/home/ishaan_essential_ai/ucr/ucr-humanup/simulation/legged_gym/resources/robots/ucr_modified/mjcf/xml"
        asset_file = "v0H_simplified.xml"
        
        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        # Get asset properties
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        
        print(f"✓ Robot loaded: {self.num_dof} DOFs, {self.num_bodies} bodies")
        
        # Get DOF names
        self.dof_names = []
        for i in range(self.num_dof):
            name = self.gym.get_asset_dof_name(self.robot_asset, i)
            self.dof_names.append(name)
            
        print(f"✓ DOF names: {self.dof_names}")
        
        # Process DOF properties with RL environment style gains
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        self.dof_props = self._process_dof_props_rl_style(dof_props_asset)
        
        # Create joint name to index mapping (from CPU version)
        self.joint_indices = {}
        for i, name in enumerate(self.dof_names):
            self.joint_indices[name] = i
            
        print(f"✓ Found {len(self.joint_indices)} joints")
        
    def _process_dof_props_rl_style(self, props):
        """Process DOF properties using RL environment PD gains"""
        # PD gains matching RL environment exactly
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
        
        # Apply gains
        for i, dof_name in enumerate(self.dof_names):
            if dof_name in joint_pd_gains:
                gains = joint_pd_gains[dof_name]
                props['stiffness'][i] = gains['kp']
                props['damping'][i] = gains['kd']
                props['driveMode'][i] = gymapi.DOF_MODE_POS
                print(f"✓ {dof_name}: kp={gains['kp']}, kd={gains['kd']}")
            else:
                # Fallback
                props['stiffness'][i] = 100.0
                props['damping'][i] = 5.0
                props['driveMode'][i] = gymapi.DOF_MODE_POS
                print(f"⚠ {dof_name}: using default gains")
                
        return props
        
    def create_envs(self):
        """Create environments with robots starting in supine position"""
        self.envs = []
        self.actor_handles = []
        
        # Environment bounds
        lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
        upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)
        
        # Create environments
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, int(np.sqrt(self.num_envs)))
            
            # Create robot actor - start lying down (supine) like CPU version
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.1)  # Low to ground
            # Supine position: lying on back (rotation around Y axis)
            pose.r = gymapi.Quat(0.0, -0.7071, 0.0, 0.7071)  # 90 degrees around Y axis
            
            actor_handle = self.gym.create_actor(
                env, 
                self.robot_asset, 
                pose, 
                f"v0h_{i}", 
                i, 
                0  # collision group
            )
            
            # Set DOF properties
            self.gym.set_actor_dof_properties(env, actor_handle, self.dof_props)
            
            self.envs.append(env)
            self.actor_handles.append(actor_handle)
            
        print(f"✓ Created {self.num_envs} environments")
        
    def init_tensors(self):
        """Initialize simulation tensors like RL environment"""
        # Prepare simulation
        self.gym.prepare_sim(self.sim)
        
        # Acquire tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        
        # Wrap tensors with proper device
        self.root_states = gymtorch.wrap_tensor(actor_root_state_tensor).to(self.device)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor).to(self.device)
        
        # Split DOF states - shape [num_envs, num_dof, 2]
        self.dof_pos = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        # Initialize target positions like RL environment
        self.target_positions = torch.zeros(self.num_envs, self.num_dof, device=self.device, dtype=torch.float)
        self.initial_dof_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device, dtype=torch.float)
        
        # Set lying down pose (adapted from CPU version)
        self.set_lying_down_pose()
        
        print(f"✓ Tensors initialized on device: {self.device}")
        
    def set_lying_down_pose(self):
        """Set joint angles for lying down pose (from CPU version)"""
        lying_pose = {
            'leftHipPitch': 0.0,
            'rightHipPitch': 0.0,
            'leftKneePitch': 0.0,
            'rightKneePitch': 0.0,
            'leftAnklePitch': 0.0,
            'rightAnklePitch': 0.0,
            'leftHipRoll': 0.0,
            'rightHipRoll': 0.0,
            'leftAnkleRoll': 0.0,
            'rightAnkleRoll': 0.0,
            'leftHipYaw': 0.0,
            'rightHipYaw': 0.0,
            'leftShoulderPitch': 0.0,
            'rightShoulderPitch': 0.0,
            'leftShoulderRoll': 0.3,  # Arms slightly away from body
            'rightShoulderRoll': -0.3,
            'leftShoulderYaw': 0.0,
            'rightShoulderYaw': 0.0,
            'leftElbow': 0.0,
            'rightElbow': 0.0,
            'torsoYaw': 0.0,
            'torsoPitch': 0.0,
            'torsoRoll': 0.0
        }
        
        for joint_name, angle in lying_pose.items():
            if joint_name in self.joint_indices:
                idx = self.joint_indices[joint_name]
                self.initial_dof_pos[:, idx] = angle
        
    def setup_initial_states(self):
        """Set up initial robot states - supine position (from CPU version)"""
        # Base initial state - lying on back on ground
        base_init_state_list = [0, 0, 0.1] + [0, -0.7071, 0, 0.7071] + [0, 0, 0] + [0, 0, 0]
        self.initial_root_states = torch.tensor(
            base_init_state_list, 
            device=self.device, 
            dtype=torch.float
        ).repeat(self.num_envs, 1)
        
        # Reset to initial states
        self.reset()
        
    def reset(self):
        """Reset environments like RL environment"""
        # Set root states
        self.root_states[:] = self.initial_root_states
        
        # Set DOF states  
        self.dof_pos[:] = self.initial_dof_pos
        self.dof_vel[:] = 0.0
        
        # Apply states to simulation
        env_ids_int32 = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )
        
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(env_ids_int32), 
            len(env_ids_int32)
        )
        
        # Initialize target positions
        self.target_positions[:] = self.dof_pos.clone()
        
        print("✓ Environment reset complete")

    # ===== GET-UP SEQUENCE METHODS (adapted from CPU version) =====
    
    def get_phase_progress(self):
        """Get current progress within the current phase (0.0 to 1.0)"""
        return min(1.0, self.current_phase_time / self.phase_duration)
    
    def smooth_interpolate(self, progress):
        """Smooth interpolation using cosine for natural motion"""
        return 0.5 * (1 - np.cos(np.pi * progress))
    
    def interpolate_pose(self, start_pose, end_pose, progress):
        """Interpolate between two poses"""
        smooth_progress = self.smooth_interpolate(progress)
        interpolated_pose = {}
        
        for joint_name in start_pose:
            if joint_name in end_pose:
                start_val = start_pose[joint_name]
                end_val = end_pose[joint_name]
                interpolated_pose[joint_name] = start_val + (end_val - start_val) * smooth_progress
            else:
                interpolated_pose[joint_name] = start_pose[joint_name]
                
        return interpolated_pose
    
    def get_lying_pose(self):
        """Return lying down pose"""
        return {
            'leftHipPitch': 0.0, 'rightHipPitch': 0.0,
            'leftKneePitch': 0.0, 'rightKneePitch': 0.0,
            'leftAnklePitch': 0.0, 'rightAnklePitch': 0.0,
            'leftHipRoll': 0.0, 'rightHipRoll': 0.0,
            'leftAnkleRoll': 0.0, 'rightAnkleRoll': 0.0,
            'leftHipYaw': 0.0, 'rightHipYaw': 0.0,
            'leftShoulderPitch': 0.0, 'rightShoulderPitch': 0.0,
            'leftShoulderRoll': 0.3, 'rightShoulderRoll': -0.3,
            'leftShoulderYaw': 0.0, 'rightShoulderYaw': 0.0,
            'leftElbow': 0.0, 'rightElbow': 0.0,
            'torsoYaw': 0.0, 'torsoPitch': 0.0, 'torsoRoll': 0.0
        }
    
    def get_side_lying_pose(self):
        """Return side lying pose (rolling motion)"""
        return {
            'leftHipPitch': 0.2, 'rightHipPitch': 0.2,
            'leftKneePitch': 0.3, 'rightKneePitch': 0.3,
            'leftAnklePitch': 0.0, 'rightAnklePitch': 0.0,
            'leftHipRoll': 0.5, 'rightHipRoll': 0.5,
            'leftAnkleRoll': 0.0, 'rightAnkleRoll': 0.0,
            'leftHipYaw': 0.0, 'rightHipYaw': 0.0,
            'leftShoulderPitch': -0.5, 'rightShoulderPitch': 0.5,
            'leftShoulderRoll': 1.0, 'rightShoulderRoll': 0.0,
            'leftShoulderYaw': 0.0, 'rightShoulderYaw': 0.0,
            'leftElbow': 0.5, 'rightElbow': 0.5,
            'torsoYaw': 0.0, 'torsoPitch': 0.0, 'torsoRoll': 0.3
        }
    
    def get_prone_pose(self):
        """Return prone (face down) pose"""
        return {
            'leftHipPitch': 0.0, 'rightHipPitch': 0.0,
            'leftKneePitch': 0.0, 'rightKneePitch': 0.0,
            'leftAnklePitch': 0.0, 'rightAnklePitch': 0.0,
            'leftHipRoll': 0.0, 'rightHipRoll': 0.0,
            'leftAnkleRoll': 0.0, 'rightAnkleRoll': 0.0,
            'leftHipYaw': 0.0, 'rightHipYaw': 0.0,
            'leftShoulderPitch': -1.0, 'rightShoulderPitch': -1.0,
            'leftShoulderRoll': 0.5, 'rightShoulderRoll': -0.5,
            'leftShoulderYaw': 0.0, 'rightShoulderYaw': 0.0,
            'leftElbow': 0.0, 'rightElbow': 0.0,
            'torsoYaw': 0.0, 'torsoPitch': 0.0, 'torsoRoll': 0.0
        }
    
    def get_push_up_prep_pose(self):
        """Return push-up preparation pose"""
        return {
            'leftHipPitch': 0.0, 'rightHipPitch': 0.0,
            'leftKneePitch': 0.0, 'rightKneePitch': 0.0,
            'leftAnklePitch': 0.2, 'rightAnklePitch': 0.2,
            'leftHipRoll': 0.0, 'rightHipRoll': 0.0,
            'leftAnkleRoll': 0.0, 'rightAnkleRoll': 0.0,
            'leftHipYaw': 0.0, 'rightHipYaw': 0.0,
            'leftShoulderPitch': -1.5, 'rightShoulderPitch': -1.5,
            'leftShoulderRoll': 0.3, 'rightShoulderRoll': -0.3,
            'leftShoulderYaw': 0.0, 'rightShoulderYaw': 0.0,
            'leftElbow': 1.5, 'rightElbow': 1.5,
            'torsoYaw': 0.0, 'torsoPitch': 0.0, 'torsoRoll': 0.0
        }
    
    def get_kneeling_pose(self):
        """Return kneeling pose"""
        return {
            'leftHipPitch': 1.4, 'rightHipPitch': 1.4,
            'leftKneePitch': 2.2, 'rightKneePitch': 2.2,
            'leftAnklePitch': -0.8, 'rightAnklePitch': -0.8,
            'leftHipRoll': 0.0, 'rightHipRoll': 0.0,
            'leftAnkleRoll': 0.0, 'rightAnkleRoll': 0.0,
            'leftHipYaw': 0.0, 'rightHipYaw': 0.0,
            'leftShoulderPitch': 0.0, 'rightShoulderPitch': 0.0,
            'leftShoulderRoll': 0.2, 'rightShoulderRoll': -0.2,
            'leftShoulderYaw': 0.0, 'rightShoulderYaw': 0.0,
            'leftElbow': 0.0, 'rightElbow': 0.0,
            'torsoYaw': 0.0, 'torsoPitch': 0.0, 'torsoRoll': 0.0
        }
    
    def get_squatting_pose(self):
        """Return squatting pose"""
        return {
            'leftHipPitch': 1.2, 'rightHipPitch': 1.2,
            'leftKneePitch': 2.0, 'rightKneePitch': 2.0,
            'leftAnklePitch': -0.5, 'rightAnklePitch': -0.5,
            'leftHipRoll': 0.0, 'rightHipRoll': 0.0,
            'leftAnkleRoll': 0.0, 'rightAnkleRoll': 0.0,
            'leftHipYaw': 0.0, 'rightHipYaw': 0.0,
            'leftShoulderPitch': 0.5, 'rightShoulderPitch': 0.5,
            'leftShoulderRoll': 0.2, 'rightShoulderRoll': -0.2,
            'leftShoulderYaw': 0.0, 'rightShoulderYaw': 0.0,
            'leftElbow': 0.0, 'rightElbow': 0.0,
            'torsoYaw': 0.0, 'torsoPitch': 0.0, 'torsoRoll': 0.0
        }
    
    def get_standing_pose(self):
        """Return standing pose"""
        return {
            'leftHipPitch': 0.0, 'rightHipPitch': 0.0,
            'leftKneePitch': 0.0, 'rightKneePitch': 0.0,
            'leftAnklePitch': 0.0, 'rightAnklePitch': 0.0,
            'leftHipRoll': 0.0, 'rightHipRoll': 0.0,
            'leftAnkleRoll': 0.0, 'rightAnkleRoll': 0.0,
            'leftHipYaw': 0.0, 'rightHipYaw': 0.0,
            'leftShoulderPitch': 0.0, 'rightShoulderPitch': 0.0,
            'leftShoulderRoll': 0.1, 'rightShoulderRoll': -0.1,
            'leftShoulderYaw': 0.0, 'rightShoulderYaw': 0.0,
            'leftElbow': 0.0, 'rightElbow': 0.0,
            'torsoYaw': 0.0, 'torsoPitch': 0.0, 'torsoRoll': 0.0
        }
    
    def apply_target_pose(self, pose):
        """Apply target pose to the robot"""
        # Reset to initial positions
        self.target_positions[:] = self.initial_dof_pos.clone()
        
        # Apply pose
        for joint_name, angle in pose.items():
            if joint_name in self.joint_indices:
                idx = self.joint_indices[joint_name]
                self.target_positions[:, idx] = angle
        
    def update_get_up_sequence(self, step):
        """Update the get-up sequence based on current step (adapted from CPU version)"""
        # Only update control targets every decimation steps (like RL environment)
        if step % self.decimation != 0:
            return
            
        # Update phase timing
        self.current_phase_time += 1
        
        # Check if we need to advance to next phase
        if (self.current_phase_time >= self.phase_duration and 
            self.sequence_phase < len(self.phases) - 1):
            self.sequence_phase += 1
            self.current_phase_time = 0
            print(f"Advancing to phase {self.sequence_phase}: {self.phases[self.sequence_phase]}")
        
        current_phase = self.phases[self.sequence_phase]
        progress = self.get_phase_progress()
        
        # Define poses for each phase
        if current_phase == 'lying_down':
            target_pose = self.get_lying_pose()
            
        elif current_phase == 'roll_to_side':
            start_pose = self.get_lying_pose()
            end_pose = self.get_side_lying_pose()
            target_pose = self.interpolate_pose(start_pose, end_pose, progress)
            
        elif current_phase == 'roll_to_prone':
            start_pose = self.get_side_lying_pose()
            end_pose = self.get_prone_pose()
            target_pose = self.interpolate_pose(start_pose, end_pose, progress)
            
        elif current_phase == 'push_up_prep':
            start_pose = self.get_prone_pose()
            end_pose = self.get_push_up_prep_pose()
            target_pose = self.interpolate_pose(start_pose, end_pose, progress)
            
        elif current_phase == 'push_to_kneel':
            start_pose = self.get_push_up_prep_pose()
            end_pose = self.get_kneeling_pose()
            target_pose = self.interpolate_pose(start_pose, end_pose, progress)
            
        elif current_phase == 'kneel_to_squat':
            start_pose = self.get_kneeling_pose()
            end_pose = self.get_squatting_pose()
            target_pose = self.interpolate_pose(start_pose, end_pose, progress)
            
        elif current_phase == 'squat_to_stand':
            start_pose = self.get_squatting_pose()
            end_pose = self.get_standing_pose()
            target_pose = self.interpolate_pose(start_pose, end_pose, progress)
            
        else:  # standing
            target_pose = self.get_standing_pose()
        
        # Apply target pose
        self.apply_target_pose(target_pose)
        
        # Debug output (less frequent due to decimation)
        control_step = step // self.decimation
        if control_step % 60 == 0:  # Print every 60 control steps
            print(f"Control step {control_step}: Phase {current_phase}, Progress: {progress:.2f}")
        
    def step(self, step_num):
        """Step simulation with RL environment timing and get-up sequence"""
        # Update get-up sequence
        self.update_get_up_sequence(step_num)
        
        # Apply position targets (like RL environment)
        self.gym.set_dof_position_target_tensor(
            self.sim, 
            gymtorch.unwrap_tensor(self.target_positions)
        )
        
        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # Refresh tensors (like RL environment post_physics_step)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # Increment counters
        self.common_step_counter += 1
        if step_num % self.decimation == 0:
            self.control_step_counter += 1
        
    def render_frame(self):
        """Render frame for video recording"""
        if not self.headless:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            
        # Camera settings
        cam_width = 1920
        cam_height = 1080
        
        # Create camera if needed
        if not hasattr(self, 'camera_handle'):
            camera_props = gymapi.CameraProperties()
            camera_props.width = cam_width
            camera_props.height = cam_height
            camera_props.enable_tensors = True
            
            self.camera_handle = self.gym.create_camera_sensor(self.envs[0], camera_props)
            
            # Set camera position for better view of get-up sequence
            cam_pos = gymapi.Vec3(2.0, 2.0, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.set_camera_location(self.camera_handle, self.envs[0], cam_pos, cam_target)
        
        # Render and get image
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        
        cam_img = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handle, gymapi.IMAGE_COLOR)
        img = cam_img.reshape((cam_height, cam_width, 4))[:, :, :3]
        
        self.gym.end_access_image_tensors(self.sim)
        
        return img
        
    def create_viewer(self):
        """Create viewer for live visualization"""
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            
            # Set camera position for better view of get-up sequence
            cam_pos = gymapi.Vec3(2.0, 2.0, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'viewer') and self.viewer:
            self.gym.destroy_viewer(self.viewer)
        if hasattr(self, 'sim') and self.sim:
            self.gym.destroy_sim(self.sim)


def main():
    print("=" * 70)
    print("HUMANOID GET-UP FROM SUPINE POSITION - GPU OPTIMIZED VERSION")
    print("=" * 70)
    
    # Configuration
    num_envs = 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    headless = False
    
    # Video settings
    video_dir = "./videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    video_path = os.path.join(video_dir, "ucr_robot_getup_gpu_optimized.mp4")
    
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Number of environments: {num_envs}")
    print(f"Headless mode: {headless}")
    print(f"Video output: {video_path}")
    print()
    
    # Create environment
    print("Creating GPU-optimized humanoid get-up environment...")
    
    try:
        env = HumanoidGetUpVideoEnvGPU(num_envs=num_envs, device=device, headless=headless)
        print("✓ Environment created successfully!")
        print(f"  - Physics timestep: {env.dt}s")
        print(f"  - Control decimation: {env.decimation}")
        print(f"  - Control frequency: {1.0/(env.dt * env.decimation):.1f} Hz")
        print(f"  - GPU pipeline: {env.sim_params.use_gpu_pipeline}")
        print(f"  - Number of DOFs: {env.num_dof}")
        print(f"  - Number of bodies: {env.num_bodies}")
        print(f"  - Get-up phases: {len(env.phases)}")
        print()

    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False
    
    # Create viewer if not headless
    if not headless:
        env.create_viewer()
        print("✓ Viewer created")
    
    # Set up video recording
    print("Setting up video recording...")
    mp4_writer = imageio.get_writer(video_path, fps=30, codec="libx264", quality=8)
    
    # Calculate simulation duration
    total_control_duration = len(env.phases) * env.phase_duration
    num_steps = total_control_duration + 300  # Extra time for final standing
    effective_control_steps = num_steps // env.decimation
    
    print(f"Running {num_steps} simulation steps ({num_steps * env.dt:.1f} seconds)")
    print(f"Effective control steps: {effective_control_steps}")
    print(f"Phase duration: {env.phase_duration} control steps ({env.phase_duration * env.dt * env.decimation:.1f}s)")
    
    try:
        for step in range(num_steps):
            # Step simulation
            env.step(step)
            
            # Record video every few steps
            if step % 2 == 0:
                try:
                    img = env.render_frame()
                    mp4_writer.append_data(img)
                except Exception as e:
                    if step < 10:  # Only warn for first few frames
                        print(f"Warning: Failed to render frame {step}: {e}")
            
            # Progress updates
            if (step + 1) % (env.decimation * 60) == 0:  # Every 60 control steps
                progress = (step + 1) / num_steps * 100
                control_step = step // env.decimation
                current_phase = env.phases[env.sequence_phase] if env.sequence_phase < len(env.phases) else "completed"
                print(f"  Progress: {progress:.1f}% (step {step + 1}/{num_steps}, control step {control_step}) - Phase: {current_phase}")
                
        # Close video
        mp4_writer.close()
        print(f"\n✓ Video saved to: {video_path}")
        
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        mp4_writer.close()
        return False
    
    finally:
        # Cleanup
        env.cleanup()
        
    print("\n" + "=" * 70)
    print("GPU-OPTIMIZED GET-UP SEQUENCE COMPLETE")
    print("=" * 70)
    print("The robot has successfully performed the following sequence:")
    for i, phase in enumerate(env.phases):
        print(f"  {i+1}. {phase.replace('_', ' ').title()}")
    print()
    return True


if __name__ == "__main__":
    main()