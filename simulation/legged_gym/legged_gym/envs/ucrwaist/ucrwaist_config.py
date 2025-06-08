# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# This file was modified by HumanUP authors in 2024-2025
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: # Copyright (c) 2021 ETH Zurich, Nikita Rudin. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024-2025 RoboVision Lab, UIUC. All rights reserved.

from legged_gym.envs.base.humanoid_config import HumanoidCfg, HumanoidCfgPPO

class V0HHumanoidCfg(HumanoidCfg):
    class env(HumanoidCfg.env):
        # use the same parallelism as G1
        num_envs = 4096
        num_actions = 23  # v0H also has 23 actuated joints (no wrists)
        n_priv = 0
        n_proprio = 3 + 2 + 3 * num_actions
        n_priv_latent = 4 + 1 + 2 * num_actions + 3
        history_len = 10

        num_observations = n_proprio + n_priv_latent + history_len * n_proprio + n_priv
        num_privileged_obs = None

        env_spacing = 3.0
        send_timeouts = True
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
        # starting position (z = 0.95 m above ground to match v0H’s pelvis pos)
        pos = [0, 0, 0.95]
        default_joint_angles = {
        # Order matches MJCF action space exactly:
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
        "rightHipPitch": -0.1,
        "rightKneePitch": 0.3,
        "rightAnklePitch": -0.2,
        "rightAnkleRoll": 0.0,
        
        # Index 17-22: Left leg
        "leftHipYaw": 0.0,
        "leftHipRoll": 0.0,
        "leftHipPitch": -0.1,
        "leftKneePitch": 0.3,
        "leftAnklePitch": -0.2,
        "leftAnkleRoll": 0.0,
    }

    class control(HumanoidCfg.control):
        stiffness = {
            "Hip": 150,
            "Knee": 200,
            "Ankle": 20,
            "Shoulder": 40,
            "Elbow": 40,
            "torso": 200,  # v0H’s torso joints are “torsoYaw,” “torsoPitch,” “torsoRoll”
        }
        damping = {
            "Hip": 5,
            "Knee": 5,
            "Ankle": 4,
            "Shoulder": 10,
            "Elbow": 10,
            "torso": 5,
        }
        action_scale = 0.5
        decimation = 20

    class sim(HumanoidCfg.sim):
        dt = 0.001
        gravity = [0, 0, -9.81]

    class normalization(HumanoidCfg.normalization):
        clip_actions = 5

    class asset(HumanoidCfg.asset):
        # point this to your v0H MJCF (or URDF if you converted it)
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/ucr_modified/mjcf/xml/v0H_merged.xml"

        # v0H’s main “base” bodies/links:
        torso_name: str = "torso"            # the actual “torso” mesh body
        chest_name: str = "torso"            # chest and torso collapse to the same link
        forehead_name: str = None            # v0H has no separate “head” collision or joint

        waist_name: str = "torsoYawLink"     # v0H’s trunk yaw joint

        # Lower‐body link names (we pick the “roll” link as thigh)
        thigh_name: str = "rightHipRollLink"  # any “HipRollLink” will denote the thigh segment
        shank_name: str = "rightKneePitchLink"  # the “knee”‐pitch link
        foot_name: str = "rightFoot"          # the final foot body containing the full mesh

        # Upper‐body link names
        upper_arm_name: str = "rightShoulderRollLink"
        lower_arm_name: str = "rightElbowLink"
        hand_name: str = None                 # v0H uses no wrist or hand meshes separately

        # Joint naming conventions (must match v0H’s <joint name="…"> tags)
        hip_name: str = "Hip"
        hip_roll_name: str = "rightHipRoll"
        hip_yaw_name: str = "rightHipYaw"
        hip_pitch_name: str = "rightHipPitch"
        knee_name: str = "rightKneePitch"
        ankle_name: str = "rightAnkleRoll"
        ankle_pitch_name: str = "rightAnklePitch"
        shoulder_name: str = "Shoulder"
        shoulder_pitch_name: str = "rightShoulderPitch"
        shoulder_roll_name: str = "rightShoulderRoll"
        shoulder_yaw_name: str = "rightShoulderYaw"
        elbow_name: str = "rightElbow"

        # feet_bodies: these are used for “feet contact” checks
        feet_bodies = ["leftFoot", "rightFoot"]
        n_lower_body_dofs: int = 12

        # we no longer penalize contacts on torso/shoulder/hip (v0H-specific)
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
            head_height_exp = 5
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

            # smooth reward
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

        base_height_target = 0.728   # similar to G1
        head_height_target = 1.3     # v0H’s head is not modeled, but keep for consistency
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
        termination_height = 0.0

    class domain_rand:
        drag_robot_up = False
        drag_robot_by_force = True
        drag_robot_part = "torso"  # drag torso upward (v0H body part)
        drag_force = 1500
        drag_force_curriculum = True
        drag_force_curriculum_type = "sin"
        drag_force_curriculum_target_height = 0.728
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
        randomize_gains = True
        stiffness_multiplier_range = 0.9  # ±90% variation
        damping_multiplier_range = 0.3

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
        experiment_name = "v0H_test"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

    class policy(HumanoidCfgPPO.policy):
        # same pattern (23 joints, no wrists)
        action_std = [0.3, 0.3, 0.3, 0.4, 0.2, 0.2] * 2 + [0.1] * 3 + [0.2] * 8
        init_noise_std = 1.0

    class algorithm(HumanoidCfgPPO.algorithm):
        learning_rate = 1e-4
        max_grad_norm = 0.5
        desired_kl = 0.005
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
