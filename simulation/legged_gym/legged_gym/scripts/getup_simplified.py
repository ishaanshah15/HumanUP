#!/usr/bin/env python3

"""
Script that instantiates the RL environment but uses hardcoded actions 
from the validation script to drive the robot through a get-up sequence.
Updated for v0H robot with new joint naming and ordering.
"""

import numpy as np
import os

import imageio
import math
from typing import Dict

# Import the RL environment and config
from legged_gym.envs.ucrwaistroll.ucrwaistroll_up import V0HWaistRollHumanUP
from legged_gym.envs.ucrwaistroll.ucrwaistroll_up_config import V0HWaistRollHumanUPCfg

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch

class HardcodedMotionController:
    """
    Controller that generates hardcoded get-up motion sequences
    adapted for the v0H robot with updated joint naming and ordering.
    """
    
    def __init__(self, num_envs: int, device: str, decimation: int = 1):
        self.num_envs = num_envs
        self.device = device
        self.decimation = decimation
        
        # Phase management
        self.sequence_phase = 0
        self.phase_duration = 180  # control steps (not simulation steps)
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
        
        # Joint name to index mapping (matches v0H actuator ordering from MJCF)
        # Order: Left leg (6) + Right leg (6) + Waist (3) + Left arm (4) + Right arm (4) = 23 DOF
        self.joint_indices = {
            # Left leg (indices 0-5)
            'left_hip_pitch_joint': 0,
            'left_hip_roll_joint': 1,
            'left_hip_yaw_joint': 2,
            'left_knee_joint': 3,
            'left_ankle_pitch_joint': 4,
            'left_ankle_roll_joint': 5,
            # Right leg (indices 6-11)
            'right_hip_pitch_joint': 6,
            'right_hip_roll_joint': 7,
            'right_hip_yaw_joint': 8,
            'right_knee_joint': 9,
            'right_ankle_pitch_joint': 10,
            'right_ankle_roll_joint': 11,
            # Waist (indices 12-14)
            'waist_yaw_joint': 12,
            'waist_roll_joint': 13,
            'waist_pitch_joint': 14,
            # Left arm (indices 15-18)
            'left_shoulder_pitch_joint': 15,
            'left_shoulder_roll_joint': 16,
            'left_shoulder_yaw_joint': 17,
            'left_elbow_joint': 18,
            # Right arm (indices 19-22)
            'right_shoulder_pitch_joint': 19,
            'right_shoulder_roll_joint': 20,
            'right_shoulder_yaw_joint': 21,
            'right_elbow_joint': 22,
        }
        
        print(f"Hardcoded controller initialized for {num_envs} envs")
        print(f"Total phases: {len(self.phases)}")
        print(f"Phase duration: {self.phase_duration} control steps")
        print(f"Total DOF: {len(self.joint_indices)}")
        
    def get_phase_progress(self):
        """Get current progress within the current phase (0.0 to 1.0)"""
        return min(1.0, self.current_phase_time / self.phase_duration)
    
    def smooth_interpolate(self, progress):
        """Smooth interpolation using cosine for natural motion"""
        return 0.5 * (1 - np.cos(np.pi * progress))
    
    def interpolate_pose(self, start_pose: Dict, end_pose: Dict, progress: float) -> Dict:
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
    
    def get_lying_pose(self) -> Dict:
        """Return lying down pose - Updated for v0H joint names"""
        return {
            # Legs
            'left_hip_pitch_joint': 0.0, 'right_hip_pitch_joint': 0.0,
            'left_knee_joint': 0.0, 'right_knee_joint': 0.0,
            'left_ankle_pitch_joint': 0.0, 'right_ankle_pitch_joint': 0.0,
            'left_hip_roll_joint': 0.0, 'right_hip_roll_joint': 0.0,
            'left_ankle_roll_joint': 0.0, 'right_ankle_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0, 'right_hip_yaw_joint': 0.0,
            # Arms
            'left_shoulder_pitch_joint': 0.0, 'right_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0, 'right_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0, 'right_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.0, 'right_elbow_joint': 0.0,
            # Waist
            'waist_yaw_joint': 0.0, 'waist_pitch_joint': 0.0, 'waist_roll_joint': 0.0
        }
    
    def get_side_lying_pose(self) -> Dict:
        """Return side lying pose (rolling motion) - Updated for v0H joint names"""
        return {
            # Legs
            'left_hip_pitch_joint': 0.2, 'right_hip_pitch_joint': 0.2,
            'left_knee_joint': 0.3, 'right_knee_joint': 0.3,
            'left_ankle_pitch_joint': 0.0, 'right_ankle_pitch_joint': 0.0,
            'left_hip_roll_joint': 0.5, 'right_hip_roll_joint': 0.5,
            'left_ankle_roll_joint': 0.0, 'right_ankle_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0, 'right_hip_yaw_joint': 0.0,
            # Arms
            'left_shoulder_pitch_joint': -0.5, 'right_shoulder_pitch_joint': 0.5,
            'left_shoulder_roll_joint': 1.0, 'right_shoulder_roll_joint': -1.0,
            'left_shoulder_yaw_joint': 0.0, 'right_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.5, 'right_elbow_joint': 0.5,
            # Waist
            'waist_yaw_joint': 0.0, 'waist_pitch_joint': 0.0, 'waist_roll_joint': 0.3
        }
    
    def get_prone_pose(self) -> Dict:
        """Return prone (face down) pose - Updated for v0H joint names"""
        return {
            # Legs
            'left_hip_pitch_joint': 0.0, 'right_hip_pitch_joint': 0.0,
            'left_knee_joint': 0.0, 'right_knee_joint': 0.0,
            'left_ankle_pitch_joint': 0.0, 'right_ankle_pitch_joint': 0.0,
            'left_hip_roll_joint': 0.0, 'right_hip_roll_joint': 0.0,
            'left_ankle_roll_joint': 0.0, 'right_ankle_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0, 'right_hip_yaw_joint': 0.0,
            # Arms
            'left_shoulder_pitch_joint': -1.0, 'right_shoulder_pitch_joint': -1.0,
            'left_shoulder_roll_joint': 0.5, 'right_shoulder_roll_joint': -0.5,
            'left_shoulder_yaw_joint': 0.0, 'right_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.0, 'right_elbow_joint': 0.0,
            # Waist
            'waist_yaw_joint': 0.0, 'waist_pitch_joint': 0.0, 'waist_roll_joint': 0.0
        }
    
    def get_push_up_prep_pose(self) -> Dict:
        """Return push-up preparation pose - Updated for v0H joint names"""
        return {
            # Legs
            'left_hip_pitch_joint': 0.0, 'right_hip_pitch_joint': 0.0,
            'left_knee_joint': 0.0, 'right_knee_joint': 0.0,
            'left_ankle_pitch_joint': 0.2, 'right_ankle_pitch_joint': 0.2,
            'left_hip_roll_joint': 0.0, 'right_hip_roll_joint': 0.0,
            'left_ankle_roll_joint': 0.0, 'right_ankle_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0, 'right_hip_yaw_joint': 0.0,
            # Arms
            'left_shoulder_pitch_joint': -1.5, 'right_shoulder_pitch_joint': -1.5,
            'left_shoulder_roll_joint': 0.3, 'right_shoulder_roll_joint': -0.3,
            'left_shoulder_yaw_joint': 0.0, 'right_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 1.5, 'right_elbow_joint': 1.5,
            # Waist
            'waist_yaw_joint': 0.0, 'waist_pitch_joint': 0.0, 'waist_roll_joint': 0.0
        }
    
    def get_kneeling_pose(self) -> Dict:
        """Return kneeling pose - Updated for v0H joint names"""
        return {
            # Legs
            'left_hip_pitch_joint': 1.4, 'right_hip_pitch_joint': 1.4,
            'left_knee_joint': 2.2, 'right_knee_joint': 2.2,
            'left_ankle_pitch_joint': -0.8, 'right_ankle_pitch_joint': -0.8,
            'left_hip_roll_joint': 0.0, 'right_hip_roll_joint': 0.0,
            'left_ankle_roll_joint': 0.0, 'right_ankle_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0, 'right_hip_yaw_joint': 0.0,
            # Arms
            'left_shoulder_pitch_joint': 0.0, 'right_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.2, 'right_shoulder_roll_joint': -0.2,
            'left_shoulder_yaw_joint': 0.0, 'right_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.0, 'right_elbow_joint': 0.0,
            # Waist
            'waist_yaw_joint': 0.0, 'waist_pitch_joint': 0.0, 'waist_roll_joint': 0.0
        }
    
    def get_squatting_pose(self) -> Dict:
        """Return squatting pose - Updated for v0H joint names"""
        return {
            # Legs
            'left_hip_pitch_joint': 0.0, 'right_hip_pitch_joint': 0.0,
            'left_knee_joint': 1.0, 'right_knee_joint': 1.0,
            'left_ankle_pitch_joint': 0.0, 'right_ankle_pitch_joint': 0.0,
            'left_hip_roll_joint': 0.0, 'right_hip_roll_joint': 0.0,
            'left_ankle_roll_joint': 0.0, 'right_ankle_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0, 'right_hip_yaw_joint': 0.0,
            # Arms
            'left_shoulder_pitch_joint': 0.0, 'right_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0, 'right_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0, 'right_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.0, 'right_elbow_joint': 0.0,
            # Waist
            'waist_yaw_joint': 0.0, 'waist_pitch_joint': 0.0, 'waist_roll_joint': 0.0
        }
    
    def get_standing_pose(self) -> Dict:
        """Return standing pose - Updated for v0H joint names"""
        return {
            # Legs
            'left_hip_pitch_joint': 0.0, 'right_hip_pitch_joint': 0.0,
            'left_knee_joint': 0.0, 'right_knee_joint': 0.0,
            'left_ankle_pitch_joint': 0.0, 'right_ankle_pitch_joint': 0.0,
            'left_hip_roll_joint': 0.0, 'right_hip_roll_joint': 0.0,
            'left_ankle_roll_joint': 0.0, 'right_ankle_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0, 'right_hip_yaw_joint': 0.0,
            # Arms
            'left_shoulder_pitch_joint': 0.0, 'right_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.1, 'right_shoulder_roll_joint': -0.1,
            'left_shoulder_yaw_joint': 0.0, 'right_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.0, 'right_elbow_joint': 0.0,
            # Waist
            'waist_yaw_joint': 0.0, 'waist_pitch_joint': 0.0, 'waist_roll_joint': 0.0
        }
    
    def pose_to_action_tensor(self, pose: Dict) -> torch.Tensor:
        """Convert pose dictionary to action tensor for RL environment"""
        # Initialize with zeros (23 DOFs for v0H)
        action = torch.zeros(23, device=self.device)
        
        # Apply pose values to corresponding indices
        for joint_name, angle in pose.items():
            if joint_name in self.joint_indices:
                idx = self.joint_indices[joint_name]
                action[idx] = angle
        
        # Repeat for all environments
        return action.unsqueeze(0).repeat(self.num_envs, 1)
    
    def get_current_action(self, control_step: int) -> torch.Tensor:
        """Get current action based on control step"""
        
        # Update phase timing every control step
        self.current_phase_time += 1

        
        
        # Check if we need to advance to next phase
        if (self.current_phase_time >= self.phase_duration and 
            self.sequence_phase < len(self.phases) - 1):
            self.sequence_phase += 1
            self.current_phase_time = 0
            print(f"Control step {control_step}: Advancing to phase {self.sequence_phase}: {self.phases[self.sequence_phase]}")
        
        current_phase = self.phases[self.sequence_phase]
        progress = self.get_phase_progress()

        start_pose = self.get_kneeling_pose()
        end_pose = self.get_squatting_pose()
        target_pose = self.interpolate_pose(start_pose, end_pose, progress)

        return self.pose_to_action_tensor(target_pose)
        
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
        
        # Debug output (less frequent)
        if control_step % 60 == 0:  # Print every 60 control steps
            print(f"Control step {control_step}: Phase {current_phase}, Progress: {progress:.2f}")
        
        return self.pose_to_action_tensor(target_pose)


def main():
    print("=" * 70)
    print("v0H RL ENVIRONMENT WITH HARDCODED GET-UP ACTIONS")
    print("=" * 70)
    
    # Configuration
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    headless = False  # Set to True if you don't want the viewer
    
    # Video settings
    video_dir = "./videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    video_path = os.path.join(video_dir, "test_rl_env_getup.mp4")
    
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Headless mode: {headless}")
    print(f"Video output: {video_path}")
    print()
    
    # Create modified config and environment
    print("Creating v0H RL environment with modified config...")
    cfg = V0HWaistRollHumanUPCfg()
    cfg.env.num_envs = 1  # Single environment for validation
    cfg.env.record_video = True  # Enable video recording
    cfg.viewer.debug_viz = True
    cfg.control.decimation = 1
    
    # Simulation parameters
    sim_params = gymapi.SimParams()
    sim_params.dt = cfg.sim.dt
    sim_params.substeps = 1
    sim_params.gravity = gymapi.Vec3(*cfg.sim.gravity)
    
    # Use CPU for easier debugging
    sim_params.use_gpu_pipeline = False
    physics_engine = gymapi.SIM_PHYSX
    device = "cpu"
    print("=" * 60)
    print("v0H WAIST ENVIRONMENT VALIDATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Physics Engine: PhysX")
    print(f"Number of environments: {cfg.env.num_envs}")
    print(f"Simulation dt: {cfg.sim.dt}")
    print(f"Total DOF: {cfg.env.num_actions}")
    print()
    
    # Create environment
    print("Creating v0H Waist environment...")
    env =  V0HWaistRollHumanUP(cfg, sim_params, physics_engine, device, headless=False)
    print("✓ v0H environment created")
    
    # Create hardcoded motion controller
    print("Creating hardcoded motion controller...")
    controller = HardcodedMotionController(
        num_envs=env.num_envs,
        device=device,
        decimation=cfg.control.decimation
    )
    print("✓ Motion controller created")
    print()
    
    # Set up video recording
    print("Setting up video recording...")
    mp4_writer = imageio.get_writer(video_path, fps=30, codec="libx264", quality=8)
    
    # Calculate simulation duration
    total_control_duration = len(controller.phases) * controller.phase_duration
    total_sim_steps = total_control_duration * cfg.control.decimation
    
    print(f"Running {total_sim_steps} simulation steps")
    print(f"Total control steps: {total_control_duration}")
    print(f"Phase duration: {controller.phase_duration} control steps")
    print(f"Expected duration: {total_sim_steps * cfg.sim.dt:.1f} seconds")
    print()
    
    control_step = 0
    
    for sim_step in range(0, total_sim_steps):
        # Get action from controller only on control steps
        if sim_step % cfg.control.decimation == 0:
            action = controller.get_current_action(control_step)
            control_step += 1
        
        # Step the RL environment with hardcoded action
        action = torch.tensor([[0.0,  0.0000,  0.0000,  0.000, 0.000,  0.0000,  0.000,  0.0000,
          0.0000,  0.0,  0.0,  0.0,  0.0000,  0.0000,  0.0000,  0.0000,
          0.00,  0.00,  0.0000,  0.00, 0.0,  0.0000,  0.0000]])

        action[0][11:14] = 2.0
        
        
        obs, priv_obs, rewards, dones, infos = env.step(5 * action)
        
        # Record video every few steps
        if sim_step % 5 == 0 and not headless:
            imgs = env.render_record(mode="rgb_array")
            if imgs is not None:
                for i in range(cfg.env.num_envs):
                    mp4_writer.append_data(imgs[i])
        
        # Progress updates
        if (sim_step + 1) % (cfg.control.decimation * 120) == 0:  # Every 120 control steps
            progress = (sim_step + 1) / total_sim_steps * 100
            current_phase = controller.phases[controller.sequence_phase] if controller.sequence_phase < len(controller.phases) else "completed"
            print(f"  Progress: {progress:.1f}% (sim step {sim_step + 1}/{total_sim_steps}) - Phase: {current_phase}")
            
            # Print some environment stats
            base_height = env.root_states[0, 2].item()
            print(f"    Base height: {base_height:.3f}m")

    # Close video
    mp4_writer.close()
    print(f"\n✓ Video saved to: {video_path}")


if __name__ == "__main__":
    main()