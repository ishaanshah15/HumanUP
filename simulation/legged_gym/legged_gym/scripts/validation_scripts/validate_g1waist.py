#!/usr/bin/env python3

"""
G1 Waist Environment Validation Script

This script loads the G1 waist humanoid environment and runs a basic simulation
with no-op actions to validate the URDF/MJCF loading and physics simulation.
No neural networks are involved - only empty actions to test the environment.
"""


import numpy as np
import os
import imageio
from isaacgym import gymapi
import torch

from legged_gym.envs.g1waist.g1waist_up import G1WaistHumanUP
from legged_gym.envs.g1waist.g1waist_up_config import G1WaistHumanUPCfg


def main():
    # Environment configuration
    cfg = G1WaistHumanUPCfg()
    cfg.env.num_envs = 1  # Single environment for validation
    cfg.env.record_video = True  # Enable video recording
    cfg.viewer.debug_viz = True
    
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
    print("G1 WAIST ENVIRONMENT VALIDATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Physics Engine: PhysX")
    print(f"Number of environments: {cfg.env.num_envs}")
    print(f"Simulation dt: {cfg.sim.dt}")
    print()
    
    
    # Create environment
    print("Creating G1 Waist environment...")
    env = G1WaistHumanUP(cfg, sim_params, physics_engine, device, headless=False)
    print(f"✓ Environment created successfully!")
    print(f"  - Number of DOFs: {env.num_dof}")
    print(f"  - Number of bodies: {env.num_bodies}")
    print(f"  - Action space size: {env.num_actions}")
    print()
    
    # Set up video recording
    mp4_writers = []
    if cfg.env.record_video:
        env.enable_viewer_sync = False
        video_dir = "../../logs/videos/validation"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        for i in range(cfg.env.num_envs):
            video_name = f"g1waist_validation_env{i}.mp4"
            video_path = os.path.join(video_dir, video_name)
            mp4_writer = imageio.get_writer(video_path, fps=50, codec="libx264")
            mp4_writers.append(mp4_writer)
        print(f"📹 Video recording to: {video_dir}")
    
    # Run simulation steps
    num_steps = 1000
    print(f"Running {num_steps} simulation steps with no-op actions...")
    
    for step in range(num_steps):
        # No-op actions (all zeros)
        actions = torch.zeros((cfg.env.num_envs, env.num_actions), device=device)
        
        # Step the environment
        obs, _, rewards, dones, infos = env.step(actions)
        
        # Record video frames
        if cfg.env.record_video:
            imgs = env.render_record(mode="rgb_array")
            if imgs is not None:
                for i in range(cfg.env.num_envs):
                    mp4_writers[i].append_data(imgs[i])
        
        # Print status every 100 steps
        if (step + 1) % 100 == 0:
            base_height = env.root_states[0, 2].item()
            print(f"  Step {step + 1:4d}: Base height = {base_height:.3f}m")
            
            # Check if robot has fallen or terminated
            if dones[0]:
                print(f"  Environment reset at step {step + 1}")
    
    print(f"\n✓ Successfully completed {num_steps} simulation steps!")
    print("✓ G1 Waist environment validation PASSED")
    
    # Final robot state
    final_height = env.root_states[0, 2].item()
    final_pos = env.root_states[0, :3]
    print(f"\nFinal robot state:")
    print(f"  Position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
    print(f"  Base height: {final_height:.3f}m")
    
    # Close video writers
    if cfg.env.record_video:
        for mp4_writer in mp4_writers:
            mp4_writer.close()
        print(f"\n📹 Video saved to: {video_dir}")
        for i in range(cfg.env.num_envs):
            video_name = f"g1waist_validation_env{i}.mp4"
            print(f"  - {video_name}")
    
   
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()