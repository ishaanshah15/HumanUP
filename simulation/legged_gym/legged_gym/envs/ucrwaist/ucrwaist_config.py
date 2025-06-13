# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# This file was modified by HumanUP authors in 2024-2025
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: # Copyright (c) 2021 ETH Zurich, Nikita Rudin. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024-2025 RoboVision Lab, UIUC. All rights reserved.

from legged_gym.envs.base.humanoid_config import HumanoidCfg, HumanoidCfgPPO

class V0HHumanoidCfg(HumanoidCfg):
    class env(HumanoidCfg.env):
        # use the same parallelism as G1
        num_envs = 4096
        num_actions = 23  # v0H has 23 actuated joints (no wrists)
        n_priv = 0
        n_proprio = 3 + 2 + 3 * num_actions  # ang_vel (3) + imu (2) + 3*actions (pos, vel, last_action)
        n_priv_latent = 4 + 1 + 2 * num_actions + 3  # mass (4) + friction (1) + motor_strength (2*23) + base_lin_vel (3)
        history_len = 10

        num_observations = n_proprio + n_priv_latent + history_len * n_proprio + n_priv

        num_privileged_obs = None

        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 10

        randomize_start_pos = False
        randomize_start_yaw = False

        history_encoding = True
        contact_buf_len = 10

        normalize_obs = True

        terminate_on_velocity = True
        terminate_on_height = True

        no_symmetry_after_stand = True


    class terrain(HumanoidCfg.terrain):
        mesh_type = "plane"

    class init_state(HumanoidCfg.init_state):
        # starting position (z = 0.95 m above ground to match v0H's pelvis pos)
        pos = [0, 0, 0.95]
        default_joint_angles = {
            # Order matches V0H MuJoCo actuator order exactly:
            # Index 0-2: Torso
            "torsoYaw": 0.0,
            "torsoPitch": 0.0,
            "torsoRoll": 0.0,
            
            # Index 3-6: Right arm
            "rightShoulderPitch": 0.0,
            "rightShoulderRoll": 0.0,
            "rightShoulderYaw": 0.0,
            "rightElbow": 0.0,
            
            # Index 7-10: Left arm  
            "leftShoulderPitch": 0.0,
            "leftShoulderRoll": 0.0,
            "leftShoulderYaw": 0.0,
            "leftElbow": 0.0,
            
            # Index 11-16: Right leg
            "rightHipYaw": 0.0,
            "rightHipRoll": 0.0,
            "rightHipPitch": -0.2,
            "rightKneePitch": 0.4,
            "rightAnklePitch": -0.2,
            "rightAnkleRoll": 0.0,
            
            # Index 17-22: Left leg
            "leftHipYaw": 0.0,
            "leftHipRoll": 0.0,
            "leftHipPitch": -0.2,
            "leftKneePitch": 0.4,
            "leftAnklePitch": -0.2,
            "leftAnkleRoll": 0.0,
        }

    class control(HumanoidCfg.control):
        # Stiffness values adapted from MuJoCo model's motor configurations
        stiffness = {
            "HipYaw": 300,
            "torsoYaw": 200,
            "HipRoll": 300,
            "HipPitch": 300,
            "torsoPitch": 200,
            "torsoRoll": 200,
            "KneePitch": 300,
            "AnklePitch": 300,
            "AnkleRoll": 150,
            "ShoulderPitch": 200,
            "ShoulderRoll": 200,
            "ShoulderYaw": 200,
            "Elbow": 150
        }
        damping = {
            "HipYaw": 10,
            "torsoYaw": 6,
            "HipRoll": 10,
            "HipPitch": 10,
            "torsoPitch": 6,
            "torsoRoll": 6,
            "KneePitch": 10,
            "AnklePitch": 10,
            "AnkleRoll": 5,
            "ShoulderPitch": 6,
            "ShoulderRoll": 6,
            "ShoulderYaw": 6,
            "Elbow": 5
        }
        action_scale = 0.5
        decimation = 20

    class sim(HumanoidCfg.sim):
        dt = 0.001  # Match MuJoCo timestep
        gravity = [0, 0, -9.81]

    class normalization(HumanoidCfg.normalization):
        clip_actions = 5

    class asset(HumanoidCfg.asset):
        # Point to your V0H MJCF file
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/ucr_modified/mjcf/xml/v0H_motor.xml"

        # V0H's main body names (from MuJoCo model)
        torso_name: str = "torso"            # the actual "torso" mesh body
        chest_name: str = "torso"            # chest and torso are the same link
        forehead_name: str = "torso"         # V0H has no separate head, use torso

        waist_name: str = "torsoYawLink"     # V0H's trunk yaw joint

        # Lower-body link names
        thigh_name: str = "HipRollLink"      # any "HipRollLink" denotes the thigh segment
        shank_name: str = "KneePitchLink"    # the "knee"-pitch link
        foot_name: str = "Foot"              # the final foot body

        # Upper-body link names
        upper_arm_name: str = "ShoulderRollLink"
        lower_arm_name: str = "ElbowLink"
        hand_name: str = None                # V0H uses no wrist or hand meshes

        # Joint naming conventions (must match V0H's <joint name="…"> tags)
        hip_name: str = "Hip"
        hip_roll_name: str = "HipRoll"
        hip_yaw_name: str = "HipYaw"
        hip_pitch_name: str = "HipPitch"
        knee_name: str = "KneePitch"
        ankle_name: str = "AnkleRoll"
        ankle_pitch_name: str = "AnklePitch"
        shoulder_name: str = "Shoulder"
        shoulder_pitch_name: str = "ShoulderPitch"
        shoulder_roll_name: str = "ShoulderRoll"
        shoulder_yaw_name: str = "ShoulderYaw"
        elbow_name: str = "Elbow"

        # feet_bodies: these are used for "feet contact" checks
        feet_bodies = ["leftFoot", "rightFoot"]
        n_lower_body_dofs: int = 12

        # No contact penalties for V0H (clean up training)
        penalize_contacts_on = []  
        terminate_after_contacts_on = []  
        dof_armature = [0.0] * 23  # can be tuned if needed

    class rewards(HumanoidCfg.rewards):
        regularization_names = [
            "dof_error",
            "dof_error_upper",
            "dof_vel",
            "feet_stumble",
            "feet_contact_forces",
            "feet_height",
            "feet_height_target_error",
            "feet_height_target_error_exp",
            "stand_on_feet",
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
            "dof_pos_limits",
            "dof_torque_limits",
            "collision",
            "torque_penalty",
        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8, 2.0]
        regularization_scale_curriculum = True
        regularization_scale_curriculum_type = "step_height"
        regularization_scale_gamma = 0.0001
        regularization_scale_curriculum_iterations = 100000

        standing_scale = 1.0
        standing_scale_range = [0.1, 0.2]
        standing_scale_curriculum = True
        standing_scale_curriculum_type = "cos"
        standing_scale_gamma = 0.0001
        standing_scale_curriculum_iterations = 50000

        class scales:
            base_height_exp = 5
            head_height_exp = 5  # Uses torso height for V0H
            delta_base_height = 1
            feet_contact_forces_increase = 1
            stand_on_feet = 2.5
            orientation = -1
            body_up_exp = 0.25
            feet_height = 2.5
            feet_distance = 2
            soft_symmetry_body = -1
            soft_symmetry_waist = -1
            feet_orientation = -0.5
            foot_slip = -1

            termination = -500

            # smooth rewards
            dof_error = -0.03
            base_lin_vel = -0.1
            ang_vel = -0.1
            dof_vel = -0.0001
            action_rate = -0.1
            torques = -0.000001
            dof_pos_limits = -5
            dof_torque_limits = -0.1
            energy = -0.0001
            dof_acc = -0.0000001

        base_height_target = 0.95    # V0H pelvis height
        head_height_target = 1.4     # V0H torso top height (approximate)
        target_feet_height = 0.1
        min_dist = 0.25
        max_dist = 1.0
        max_knee_dist = 0.5
        target_joint_pos_scale = 0.17
        cycle_time = 0.64
        double_support_threshold = 0.1
        only_positive_rewards = False
        clip_inf_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 500
        max_contact_force_head = 250
        max_contact_force_torso = 250
        termination_height = 0.2  # Lower than G1 due to different robot proportions

    class domain_rand:
        drag_robot_up = False
        drag_robot_by_force = True
        drag_robot_part = "torso"  # drag torso upward (V0H body part)
        drag_force = 1500
        drag_force_curriculum = True
        drag_force_curriculum_type = "sin"
        drag_force_curriculum_target_height = 0.95
        drag_interval = 50
        drag_when_falling = False
        force_compenstation = False
        min_drag_vel = 0.1
        max_drag_vel = 0.5

        domain_rand_general = False
        randomize_gravity = True and domain_rand_general
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)

        randomize_friction = True and domain_rand_general
        friction_range = [0.6, 2.0]

        randomize_base_mass = True and domain_rand_general
        added_mass_range = [-3.0, 3]

        randomize_base_com = True and domain_rand_general
        added_com_range = [-0.05, 0.05]

        push_robots = True and domain_rand_general
        push_interval_s = 4
        max_push_vel_xy = 1.0

        randomize_motor = True and domain_rand_general
        motor_strength_range = [0.8, 1.2]

        action_delay = True and domain_rand_general
        action_buf_len = 8

    class noise(HumanoidCfg.noise):
        add_noise = False
        noise_increasing_steps = 5000

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.05
            gravity = 0.05
            imu = 0.05

    class commands:
        curriculum = False
        num_commands = 1
        resampling_time = 3.0

        ang_vel_clip = 0.1
        lin_vel_clip = 0.1

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [-0.3, 0.3]
            lin_vel_z = [0.0, 0.3]
            ang_vel_yaw = [-0.6, 0.6]


class V0HHumanoidCfgPPO(HumanoidCfgPPO):
    seed = 1

    class runner(HumanoidCfgPPO.runner):
        policy_class_name = "ActorCriticRMA"
        algorithm_class_name = "PPORMA"
        runner_class_name = "OnPolicyRunner"
        max_iterations = 50001

        save_interval = 100
        experiment_name = "v0H_humanoid"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

    class policy(HumanoidCfgPPO.policy):
        # V0H action standard deviations (23 joints)
        # [torso(3), right_arm(4), left_arm(4), right_leg(6), left_leg(6)]
        action_std = [0.2, 0.2, 0.2] + [0.3, 0.3, 0.3, 0.4] * 2 + [0.4, 0.2, 0.4, 0.4, 0.2, 0.2] * 2
        init_noise_std = 1.0

    class algorithm(HumanoidCfgPPO.algorithm):
        learning_rate = 1e-4
        max_grad_norm = 0.5
        desired_kl = 0.005
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]