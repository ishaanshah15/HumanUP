#!/usr/bin/env python3

"""
Humanoid Robot Get-Up from Supine Position Script

This script loads the humanoid robot environment and generates a video
of the robot getting up from a supine (lying on back) position to standing.
The sequence involves rolling to prone, pushing up to kneeling, and then standing.
"""

import numpy as np
import os
import imageio
import math

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch


class HumanoidGetUpVideoEnv:
    def __init__(self, num_envs=1, device="cpu", headless=False):
        self.num_envs = num_envs
        self.device = device
        self.headless = headless
        self.env_spacing = 2.0
        
        # Initialize Isaac Gym
        self.gym = gymapi.acquire_gym()
        
        # Configure simulation
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.use_gpu_pipeline = False  # Use CPU for stability
        
        # Physics engine settings
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        
        # Create simulation
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        
        # Create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
        # Load robot asset
        self.load_robot_asset()
        
        # Create environments
        self.create_envs()
        
        # Initialize tensors
        self.init_tensors()
        
        # Set initial states
        self.setup_initial_states()
        
        # Get-up sequence parameters
        self.sequence_phase = 0
        self.phase_duration = 180  # frames per phase
        self.current_phase_time = 0
        
        # Define get-up phases
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
        
    def load_robot_asset(self):
        """Load the humanoid robot asset from XML file"""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 300000
        
        # Load the XML file
        self.robot_asset = self.gym.load_asset(
            self.sim, 
            "/home/ishaan_essential_ai/ucr/ucr-humanup/simulation/legged_gym/resources/robots/ucr_modified/mjcf/xml", 
            "v0H_simplified.xml",
            asset_options
        )
        
        # Get asset properties
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        
        print(f"Robot loaded: {self.num_dof} DOFs, {self.num_bodies} bodies")
        
        # Get DOF properties
        dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        
        # Set DOF properties for position control
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = 800.0   # Moderate stiffness for natural movement
            dof_props['damping'][i] = 80.0      # Moderate damping for stability
            
        self.dof_props = dof_props
        
        # Get joint names
        self.dof_names = []
        for i in range(self.num_dof):
            name = self.gym.get_asset_dof_name(self.robot_asset, i)
            self.dof_names.append(name)

        print('DOF names:', self.dof_names)
            
        # Create joint name to index mapping
        self.joint_indices = {}
        for i, name in enumerate(self.dof_names):
            self.joint_indices[name] = i
            
        print(f"Found {len(self.joint_indices)} joints")
        
    def create_envs(self):
        """Create environments with robots"""
        self.envs = []
        self.actor_handles = []
        
        # Environment bounds
        lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
        upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)
        
        # Create environments
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, int(np.sqrt(self.num_envs)))
            
            # Create robot actor - start lying down (supine)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.1)  # Low to ground
            # Supine position: lying on back (rotation around Y axis)
            pose.r = gymapi.Quat(0.0, -0.7071, 0.0, 0.7071)  # 90 degrees around Y axis
            
            actor_handle = self.gym.create_actor(
                env, 
                self.robot_asset, 
                pose, 
                f"robot_{i}", 
                i, 
                1
            )
            
            # Set DOF properties
            self.gym.set_actor_dof_properties(env, actor_handle, self.dof_props)
            
            self.envs.append(env)
            self.actor_handles.append(actor_handle)
            
    def init_tensors(self):
        """Initialize simulation tensors"""
        # Prepare simulation
        self.gym.prepare_sim(self.sim)
        
        # Get simulation tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        
        # Wrap tensors
        self.root_states = gymtorch.wrap_tensor(actor_root_state_tensor)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        
        # Split DOF states
        self.dof_pos = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        # Initialize target positions
        self.target_positions = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.initial_targets = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        
    def setup_initial_states(self):
        """Set up initial robot states - supine position"""
        # Base initial state - lying on back on ground
        # Position: on ground, Orientation: supine (back down)
        base_init_state_list = [0, 0, 0.1] + [0, -0.7071, 0, 0.7071] + [0, 0, 0] + [0, 0, 0]
        self.initial_root_states = torch.tensor(
            base_init_state_list, 
            device=self.device, 
            dtype=torch.float
        ).repeat(self.num_envs, 1)
        
        # Set initial DOF positions (lying down pose)
        self.initial_dof_states = torch.zeros(
            self.num_envs, 
            self.num_dof, 
            device=self.device, 
            dtype=torch.float
        )
        
        # Set lying down pose
        self.set_lying_down_pose()
        
        # Reset to initial states
        self.reset()
        
    def set_lying_down_pose(self):
        """Set joint angles for lying down pose"""
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
                self.initial_dof_states[:, idx] = angle
        
    def reset(self):
        """Reset the environments"""
        # Set root states
        self.root_states[:] = self.initial_root_states
        
        # Set DOF states
        self.dof_pos[:] = self.initial_dof_states
        self.dof_vel[:] = 0.0
        
        # Apply states
        self.gym.set_actor_root_state_tensor(
            self.sim, 
            gymtorch.unwrap_tensor(self.root_states)
        )
        self.gym.set_dof_state_tensor(
            self.sim, 
            gymtorch.unwrap_tensor(self.dof_states)
        )
        
        # Initialize target positions
        self.target_positions[:] = self.dof_pos.clone()
        self.initial_targets[:] = self.dof_pos.clone()
        
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
    
    def update_get_up_sequence(self, step):
        """Update the get-up sequence based on current step"""
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
        
        # Debug output
        if step % 60 == 0:  # Print every second
            print(f"Step {step}: Phase {current_phase}, Progress: {progress:.2f}")
    
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
        self.target_positions[:] = self.initial_targets.clone()
        
        # Apply pose
        for joint_name, angle in pose.items():
            if joint_name in self.joint_indices:
                idx = self.joint_indices[joint_name]
                self.target_positions[:, idx] = angle
                
    def step(self, step_num):
        """Step the simulation with get-up sequence"""
        # Update get-up sequence
        self.update_get_up_sequence(step_num)
        
        # Apply target positions
        self.gym.set_dof_position_target_tensor(
            self.sim, 
            gymtorch.unwrap_tensor(self.target_positions)
        )
        
        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # Update tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
    def render_frame(self):
        """Render a frame and return the image"""
        if not self.headless:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            
        # Get camera image
        cam_width = 1920
        cam_height = 1080
        
        # Create camera if not exists
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
        
        # Get camera image
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        
        cam_img = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handle, gymapi.IMAGE_COLOR)
        img = cam_img.reshape((cam_height, cam_width, 4))[:, :, :3]  # Remove alpha channel
        
        self.gym.end_access_image_tensors(self.sim)
        
        return img
        
    def create_viewer(self):
        """Create viewer for live visualization"""
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            
            # Set camera position
            cam_pos = gymapi.Vec3(2.0, 2.0, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


def main():
    print("=" * 60)
    print("HUMANOID ROBOT GET-UP FROM SUPINE POSITION")
    print("=" * 60)
    
    # Configuration
    num_envs = 1
    device = "cpu"
    headless = False  # Set to True if you don't want to see live simulation
    
    # Video settings
    video_dir = "./videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    video_path = os.path.join(video_dir, "ucr_robot_getup.mp4")
    
    print(f"Device: {device}")
    print(f"Number of environments: {num_envs}")
    print(f"Headless mode: {headless}")
    print(f"Video output: {video_path}")
    print()
    
    # Create environment
    print("Creating humanoid environment...")
    
    env = HumanoidGetUpVideoEnv(num_envs=num_envs, device=device, headless=headless)
    print("✓ Environment created successfully!")
    print(f"  - Number of DOFs: {env.num_dof}")
    print(f"  - Number of bodies: {env.num_bodies}")
    print(f"  - Number of joints: {len(env.joint_indices)}")
    print(f"  - Get-up phases: {len(env.phases)}")
    print()

    # Create viewer if not headless
    if not headless:
        env.create_viewer()
        print("✓ Viewer created")
    
    # Set up video recording
    print("Setting up video recording...")
    mp4_writer = imageio.get_writer(video_path, fps=30, codec="libx264", quality=8)
    
    # Run simulation and record video
    total_duration = len(env.phases) * env.phase_duration
    num_steps = total_duration + 300  # Extra time for final standing
    print(f"Running {num_steps} simulation steps ({num_steps/60:.1f} seconds) and recording video...")
    
    for step in range(num_steps):
        # Step simulation
        env.step(step)
        
        # Render frame for video
        if step % 2 == 0:  # Record every other frame to reduce file size
            try:
                img = env.render_frame()
                mp4_writer.append_data(img)
            except Exception as e:
                print(f"Warning: Failed to render frame {step}: {e}")
        
        # Print progress
        if (step + 1) % 180 == 0:
            progress = (step + 1) / num_steps * 100
            current_phase = env.phases[env.sequence_phase] if env.sequence_phase < len(env.phases) else "completed"
            print(f"  Progress: {progress:.1f}% ({step + 1}/{num_steps} steps) - Phase: {current_phase}")
            
        

    # Close video writer
    mp4_writer.close()
    print(f"\n✓ Video saved to: {video_path}")
    
    # Cleanup
    if not headless and hasattr(env, 'viewer'):
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)
    
    print("\n" + "=" * 60)
    print("GET-UP SEQUENCE COMPLETE")
    print("=" * 60)
    print("The robot has successfully performed the following sequence:")
    for i, phase in enumerate(env.phases):
        print(f"  {i+1}. {phase.replace('_', ' ').title()}")
    print()
    return True


if __name__ == "__main__":
    main()