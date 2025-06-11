#!/usr/bin/env python3

"""
Humanoid Right Arm Lift Video Generation Script - FIXED VERSION

This script loads the humanoid robot environment and generates a video
of the robot lifting its right arm using controlled joint movements.
"""

import numpy as np
import os
import imageio

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch


class HumanoidArmLiftVideoEnv:
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
        
        # Animation parameters
        self.animation_phase = 0.0
        self.animation_speed = 0.01  # Slower for more visible movement
        
    def load_robot_asset(self):
        """Load the humanoid robot asset from URDF file"""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 300000
        
        # Load the URDF file
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
            dof_props['stiffness'][i] = 1000.0  # Higher stiffness for better control
            dof_props['damping'][i] = 100.0     # Higher damping for stability
            
        self.dof_props = dof_props
        
        # Get joint names to identify right arm joints
        self.dof_names = []
        for i in range(self.num_dof):
            name = self.gym.get_asset_dof_name(self.robot_asset, i)
            self.dof_names.append(name)

        print('DOF names:', self.dof_names)
            
        # Find right arm joint indices - CORRECTED NAMES
        self.right_arm_indices = {}
        right_arm_joints = [
            'right_shoulder_pitch_joint',
            'right_shoulder_roll_joint', 
            'right_shoulder_yaw_joint',
            'right_elbow_joint'
        ]

        right_arm_joints =  ['rightShoulderPitch', 'rightShoulderRoll', 'rightShoulderYaw', 'rightElbow']
        
        for joint_name in right_arm_joints:
            if joint_name in self.dof_names:
                idx = self.dof_names.index(joint_name)
                self.right_arm_indices[joint_name] = idx
                print(f"✓ Right arm joint {joint_name} at index {idx}")
            else:
                print(f"✗ Right arm joint {joint_name} NOT FOUND")
                
        print(f"Found {len(self.right_arm_indices)} right arm joints")
        
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
            
            # Create robot actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.8)  # Raised higher for better visibility
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # No rotation
            
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
        """Set up initial robot states"""
        # Base initial state - standing upright
        base_init_state_list = [0, 0, 0.2] + [0, -0.7071, 0, 0.7071] + [0, 0, 0] + [0, 0, 0]
        self.initial_root_states = torch.tensor(
            base_init_state_list, 
            device=self.device, 
            dtype=torch.float
        ).repeat(self.num_envs, 1)
        
        # Set initial DOF positions (neutral pose)
        self.initial_dof_states = torch.zeros(
            self.num_envs, 
            self.num_dof, 
            device=self.device, 
            dtype=torch.float
        )
        
        # Reset to initial states
        self.reset()
        
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
        
    def update_arm_animation(self, step):
        """Update right arm target positions for smooth animation"""
        # Animation progress (0 to 1)
        progress = min(1.0, step * self.animation_speed)
        
        # Smooth interpolation using cosine for natural motion
        smooth_progress = 0.5 * (1 - np.cos(np.pi * progress))
        
        # Reset to initial positions
        self.target_positions[:] = self.initial_targets.clone()
        
        # Define final target positions for right arm - ADJUSTED VALUES
        arm_targets = {
            'rightShoulderPitch': -0.8,  # Move arm forward (negative for right arm)
            'rightShoulderRoll': -1.2,   # Move arm away from body (negative for right arm)
            'rightShoulderYaw': 0.5,     # Slight rotation
            'rightElbow': 1.5              # Bend elbow significantly
        }
        
        # Interpolate between initial and target positions
        for joint_name, target_angle in arm_targets.items():
            if joint_name in self.right_arm_indices:
                idx = self.right_arm_indices[joint_name]
                initial_angle = self.initial_targets[0, idx]
                
                # Smooth interpolation
                current_target = initial_angle + (target_angle - initial_angle) * smooth_progress
                self.target_positions[:, idx] = current_target
                
                # Debug output for first few steps
                if step < 10:
                    print(f"Step {step}: {joint_name} -> target: {current_target:.3f}")
                
    def step(self, step_num):
        """Step the simulation with animated arm movement"""
        # Update arm animation
        self.update_arm_animation(step_num)
        
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
            
            # Set camera position for better view
            cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
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
            cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


def main():
    print("=" * 60)
    print("HUMANOID RIGHT ARM LIFT VIDEO GENERATION - FIXED")
    print("=" * 60)
    
    # Configuration
    num_envs = 1
    device = "cpu"
    headless = False  # Set to True if you don't want to see live simulation
    
    # Video settings
    video_dir = "./videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    video_path = os.path.join(video_dir, "ucr_arm_lift_fixed.mp4")
    
    print(f"Device: {device}")
    print(f"Number of environments: {num_envs}")
    print(f"Headless mode: {headless}")
    print(f"Video output: {video_path}")
    print()
    
    # Create environment
    print("Creating humanoid environment...")
    
    
    env = HumanoidArmLiftVideoEnv(num_envs=num_envs, device=device, headless=headless)
    print("✓ Environment created successfully!")
    print(f"  - Number of DOFs: {env.num_dof}")
    print(f"  - Number of bodies: {env.num_bodies}")
    print(f"  - Right arm joints found: {len(env.right_arm_indices)}")
    print()

    
    # Create viewer if not headless
    if not headless:
        env.create_viewer()
        print("✓ Viewer created")
    
    # Set up video recording
    print("Setting up video recording...")
    mp4_writer = imageio.get_writer(video_path, fps=30, codec="libx264", quality=8)
    
    # Run simulation and record video
    num_steps = 1000  # Longer simulation for slower movement
    print(f"Running {num_steps} simulation steps and recording video...")
    
   
    for step in range(num_steps):
        # Step simulation
        
        env.step(step)
        #print(f"Step {step + 1} out of {num_steps}")
        # Render frame for video
        if step % 2 == 0:  # Record every other frame to reduce file size
            try:
                img = env.render_frame()
                mp4_writer.append_data(img)
            except Exception as e:
                print(f"Warning: Failed to render frame {step}: {e}")
        
        
        
        # Print progress
        if (step + 1) % 100 == 0:
            progress = (step + 1) / num_steps * 100
            print(f"  Progress: {progress:.1f}% ({step + 1}/{num_steps} steps)")
            
            # Print current joint positions for debugging
            if hasattr(env, 'dof_pos') and len(env.right_arm_indices) > 0:
                print("  Current right arm positions:")
                for joint_name, idx in env.right_arm_indices.items():
                    current_pos = env.dof_pos[0, idx].item()
                    print(f"    {joint_name}: {current_pos:.3f}")

    # Close video writer
    mp4_writer.close()
    print(f"\n✓ Video saved to: {video_path}")
    
    # Cleanup
    if not headless and hasattr(env, 'viewer'):
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)
    
    print("\n" + "=" * 60)
    print("VIDEO GENERATION COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()