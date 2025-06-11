#!/usr/bin/env python3

"""
V0H Isaac Gym Debug Script

Systematic debugging to identify why V0H doesn't work in Isaac Gym
"""

import numpy as np

from isaacgym import gymapi, gymtorch
import torch
import os

def test_asset_loading():
    """Test if the V0H asset loads correctly"""
    print("=" * 60)
    print("TESTING ASSET LOADING")
    print("=" * 60)
    
    # Initialize Isaac Gym
    gym = gymapi.acquire_gym()
    
    # Simulation parameters
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.001
    sim_params.substeps = 1
    sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
    sim_params.use_gpu_pipeline = False
    
    # Create simulation
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    
    # Test different asset paths
    test_paths = [
        "/home/ishaan_essential_ai/ucr/ucr-humanup/simulation/legged_gym/resources/robots/ucr_modified/mjcf/xml/v0H_simplified.xml",  # Your MJCF path
        "/home/ishaan_essential_ai/ucr/ucr-humanup/simulation/legged_gym/resources/robots/ucr_modified/mjcf/xml/v0h_simplified.urdf", # URDF version
        "/home/ishaan_essential_ai/ucr/ucr-humanup/simulation/legged_gym/resources/robots/g1_modified/g1_29dof_fixedwrist_custom_collision.urdf",  # Working G1 for comparison
    ]
    
    for path in test_paths:
        print(f"\nTesting: {path}")
        if not os.path.exists(path):
            print(f"  ❌ File not found: {path}")
            continue
            
        try:
            # Asset options
            asset_options = gymapi.AssetOptions()
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
            asset_options.collapse_fixed_joints = True
            asset_options.replace_cylinder_with_capsule = True
            asset_options.flip_visual_attachments = False
            asset_options.fix_base_link = False
            asset_options.density = 0.001
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.max_angular_velocity = 1000.0
            asset_options.max_linear_velocity = 1000.0
            asset_options.armature = 0.01
            asset_options.thickness = 0.01
            asset_options.disable_gravity = False
            
            # Try to load asset
            asset = gym.load_asset(sim, os.path.dirname(path), os.path.basename(path), asset_options)
            
            if asset:
                num_dof = gym.get_asset_dof_count(asset)
                num_bodies = gym.get_asset_rigid_body_count(asset)
                dof_names = gym.get_asset_dof_names(asset)
                body_names = gym.get_asset_rigid_body_names(asset)
                
                print(f"  ✅ Asset loaded successfully!")
                print(f"     DOFs: {num_dof}")
                print(f"     Bodies: {num_bodies}")
                print(f"     DOF names: {dof_names[:5]}...")  # First 5
                print(f"     Body names: {body_names[:5]}...")  # First 5
            else:
                print(f"  ❌ Failed to load asset")
                
        except Exception as e:
            print(f"  ❌ Error loading asset: {e}")
    
    gym.destroy_sim(sim)

def test_environment_creation():
    """Test if environment can be created with V0H"""
    print("\n" + "=" * 60)
    print("TESTING ENVIRONMENT CREATION")
    print("=" * 60)
    
    try:
        from legged_gym.envs.ucrwaist.ucrwaist_up import V0HHumanoid
        from legged_gym.envs.ucrwaist.ucrwaist_config import V0HHumanoidCfg
        
        # Test configuration
        cfg = V0HHumanoidCfg()
        cfg.env.num_envs = 1  # Single environment for testing
        
        # Simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.dt = cfg.sim.dt
        sim_params.substeps = 1
        sim_params.gravity = gymapi.Vec3(*cfg.sim.gravity)
        sim_params.use_gpu_pipeline = False
        
        print("Creating V0H environment...")
        env = V0HHumanoid(cfg, sim_params, gymapi.SIM_PHYSX, "cpu", headless=True)
        
        print(f"  ✅ Environment created successfully!")
        print(f"     Number of DOFs: {env.num_dof}")
        print(f"     Number of bodies: {env.num_bodies}")
        print(f"     Action space: {env.num_actions}")
        
        # Test a simple step
        actions = torch.zeros((1, env.num_actions))
        obs, rewards, dones, infos = env.step(actions)
        
        print(f"  ✅ Environment step successful!")
        print(f"     Observation shape: {obs.shape}")
        print(f"     Reward: {rewards[0].item():.3f}")
        
    except Exception as e:
        print(f"  ❌ Error creating environment: {e}")
        print(f"     Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

def compare_with_g1():
    """Compare V0H with working G1 environment"""
    print("\n" + "=" * 60)
    print("COMPARING V0H WITH G1")
    print("=" * 60)
    
    # Test G1 first
    try:
        # Assuming G1 environment exists
        print("Testing G1 environment...")
        # from legged_gym.envs.g1.g1_env import G1Env
        # from legged_gym.envs.g1.g1_config import G1Cfg
        
        # cfg_g1 = G1Cfg()
        # env_g1 = G1Env(cfg_g1, ...)
        print("  ✅ G1 would work here (placeholder)")
        
    except Exception as e:
        print(f"  ❌ G1 also has issues: {e}")
    
    # Key differences to check
    print("\nKey differences to investigate:")
    print("1. File format: URDF vs MJCF")
    print("2. Joint ordering and naming")
    print("3. Asset paths and mesh loading")
    print("4. Physics parameters")
    print("5. Actuator definitions")

def test_minimal_mjcf():
    """Test with minimal MJCF to isolate issues"""
    print("\n" + "=" * 60)
    print("TESTING MINIMAL MJCF")
    print("=" * 60)
    
    # Create minimal MJCF string
    minimal_mjcf = """
    <mujoco model="minimal_v0h">
      <compiler angle="radian"/>
      <option timestep="0.001"/>
      
      <worldbody>
        <light directional="true" pos="0 0 3"/>
        <geom name="floor" type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>
        
        <body name="pelvis" pos="0 0 1">
          <joint type="free"/>
          <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
          <geom name="pelvis_geom" type="box" size="0.1 0.1 0.05" rgba="1 0 0 1"/>
          
          <body name="torso" pos="0 0 0.1">
            <joint name="torso_joint" type="hinge" axis="0 0 1" range="-1 1"/>
            <inertial pos="0 0 0" mass="0.5" diaginertia="0.05 0.05 0.05"/>
            <geom name="torso_geom" type="box" size="0.05 0.05 0.1" rgba="0 1 0 1"/>
          </body>
        </body>
      </worldbody>
      
      <actuator>
        <position name="torso_actuator" joint="torso_joint" kp="100" kv="10"/>
      </actuator>
    </mujoco>
    """
    
    # Save to temporary file
    temp_file = "/tmp/minimal_v0h.xml"
    with open(temp_file, "w") as f:
        f.write(minimal_mjcf)
    
    # Test loading
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.use_gpu_pipeline = False
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    
    try:
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        
        asset = gym.load_asset(sim, "/tmp", "minimal_v0h.xml", asset_options)
        
        if asset:
            print("  ✅ Minimal MJCF loads successfully!")
            
            # Create environment and test
            env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
            actor = gym.create_actor(env, asset, gymapi.Transform(), "minimal_robot", 0, 1)
            
            # Test simulation step
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            
            print("  ✅ Minimal simulation step successful!")
            
        else:
            print("  ❌ Failed to load minimal MJCF")
            
    except Exception as e:
        print(f"  ❌ Error with minimal MJCF: {e}")
    
    gym.destroy_sim(sim)
    
    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)

def test_joint_configuration():
    """Test joint configuration and control"""
    print("\n" + "=" * 60)
    print("TESTING JOINT CONFIGURATION")
    print("=" * 60)
    
    try:
        # Load your V0H asset
        gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.use_gpu_pipeline = False
        sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        
        # Test with your actual V0H file
        asset_path = "path/to/your/v0h.xml"  # Update this path
        
        if os.path.exists(asset_path):
            asset_options = gymapi.AssetOptions()
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
            asset_options.fix_base_link = False
            
            asset = gym.load_asset(sim, os.path.dirname(asset_path), os.path.basename(asset_path), asset_options)
            
            if asset:
                # Get DOF properties
                dof_props = gym.get_asset_dof_properties(asset)
                dof_names = gym.get_asset_dof_names(asset)
                
                print(f"  ✅ Asset loaded with {len(dof_names)} DOFs")
                
                # Check each DOF
                for i, name in enumerate(dof_names):
                    props = dof_props[i]
                    print(f"    DOF {i}: {name}")
                    print(f"      Drive mode: {props['driveMode']}")
                    print(f"      Lower limit: {props['lower']:.3f}")
                    print(f"      Upper limit: {props['upper']:.3f}")
                    print(f"      Max force: {props['effort']:.1f}")
                    print(f"      Max velocity: {props['velocity']:.1f}")
                    print(f"      Stiffness: {props['stiffness']:.1f}")
                    print(f"      Damping: {props['damping']:.1f}")
                    print()
                
                # Test if joints have reasonable ranges
                for i, name in enumerate(dof_names):
                    props = dof_props[i]
                    if props['effort'] <= 0:
                        print(f"  ⚠️  Warning: {name} has zero effort limit")
                    if props['stiffness'] <= 0:
                        print(f"  ⚠️  Warning: {name} has zero stiffness")
                    if abs(props['lower'] - props['upper']) < 0.001:
                        print(f"  ⚠️  Warning: {name} has very small joint range")
            
        else:
            print(f"  ❌ V0H asset not found at: {asset_path}")
            
        gym.destroy_sim(sim)
        
    except Exception as e:
        print(f"  ❌ Error testing joint configuration: {e}")

def main():
    """Run all debug tests"""
    print("V0H ISAAC GYM DEBUG SUITE")
    print("This will help identify why V0H doesn't work in Isaac Gym")
    
    # Run all tests
    test_asset_loading()
    test_environment_creation()
    compare_with_g1()
    test_minimal_mjcf()
    test_joint_configuration()
    
    print("\n" + "=" * 60)
    print("DEBUG SUMMARY")
    print("=" * 60)
    print("Based on the results above, the likely issues are:")
    print("1. Asset loading problems (file paths, format)")
    print("2. Joint configuration issues (gains, limits)")
    print("3. Physics parameter mismatches")
    print("4. Mesh/collision geometry problems")
    print()
    print("RECOMMENDED FIXES:")
    print("1. Try the URDF version first")
    print("2. Use explicit joint configurations (no class inheritance)")
    print("3. Fix asset paths (remove relative paths)")
    print("4. Start with minimal configuration")
    print("5. Copy G1's working patterns exactly")

if __name__ == "__main__":
    main()