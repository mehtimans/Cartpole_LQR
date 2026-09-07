# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from cartpole_sim2real.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Cartpolesim2simCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        frame_stack = 10
        c_frame_stack = 10
        num_single_obs = 5 
        num_observations = int(frame_stack * num_single_obs) # 5*10 = 50
        single_num_privileged_obs = 5 # 6  
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs) # 6 * 10 = 60
        actor_conv_layers = [1, 16, 32]
        critic_conv_layers = [1, 16, 32]
        num_actions = 1
        num_envs = 500
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm 
        episode_length_s = 25 # Changed episode_length_s from 20 to 40 on purpose.
        use_ref_actions = False
        reset_dist = 0.3 # 68 cm

    class terrain:
        mesh_type = "plane"
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = False
        static_friction = 1.0  # static_friction and dynamic_friction have no effect here since there is no contact with the ground.
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = False
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 0
        resampling_time = 10.   # Has no effect because resample_command is not used.
        heading_command = False # Has no effect
        
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {
            "slider_to_cart": 0.0, 
            "cart_to_pole": 0.0} # 0.12

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cartpole/urdf/cartpole.urdf'
        name = "cartpole"
        foot_name = ""  # Has no effect
        penalize_contacts_on = []
        terminate_after_contacts_on = ["pole"]
        disable_gravity = False
        collapse_fixed_joints = False # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = True
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        num_buckets_friction = 64
        friction_range = [0.0, 1.0] 
        friction_default = 0.0

        randomize_restitution = True
        num_buckets_restitution = 64
        restitution_range = [0.15, 0.25]
        restitution_default = 0.2
        
        randomize_joint_damping =  True
        slider_joint_damping_range = [5e-2, 4.] 
        pole_joint_damping_range = [5e-4, 5e-2]  
        default_slider_joint_damping_range = 0.01   
        default_pole_joint_damping_range = 0.005  

        randomize_joint_friction = True
        slider_joint_friction_range = [1e-3, 0.35]  
        pole_joint_friction_range = [0.0, 5e-2] 
        default_slider_joint_friction_range = 0.01   
        default_pole_joint_friction_range = 0.005  
        
        randomize_joint_stiffness = True
        slider_joint_stiffness_range = [0.001, 5.]  
        pole_joint_stiffness_range = [0.0, 0.05] 
        default_slider_joint_stiffness_range = 1.  
        default_pole_joint_stiffness_range = 0.005  

        randomize_base_mass = True
        added_cart_mass_range = [-0.1 , 0.1]
        added_pole_mass_range = [-0.02, 0.02]
        
        randomize_com_displacement = True
        cart_com_displacement_xrange = [-0.001, 0.001]
        cart_com_displacement_yrange = [-0.005, 0.005]
        cart_com_displacement_zrange = [-0.001, 0.001]
        pole_com_displacement_xrange = [-0.002, 0.002]
        pole_com_displacement_yrange = [-0.002, 0.002]
        pole_com_displacement_zrange = [-0.04, 0.04]
        
        action_delay = 0.5
        action_noise = 0.05
        
        push_force = True
        push_force_interval_s = 15
        max_push_force = 2. #0.4
         
        randomize_dof_pos = True 
        num_buckets_dof = 64
        cart_dof_range = [0.0, 0.0]
        pole_dof_range = [-0.1, 0.1]


    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'Actuator_network' # choose between P ,T and Actuator_network for now
        slider_to_cart_stiffness = 10.0  # kp [N*m/rad]
        slider_to_cart_damping = 10.0     # kd [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 15 # 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 6
        # Actuator frame stack: number of past steps used to store error and velocity
        # for constructing the actuator network input
        actuator_frame_stack = 21
        pulley_radius = 0.025490 # [m]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        class scales(LeggedRobotCfg.rewards.scales):
            remain_up = 0.
            vel_tracking = 1.5
            decay = 0.
            torques = -0.5
            balance = 0.4
            torque_limits = -0.01
            terminations = -0.0001
            action_rate = -0.05
            cart_acc = -0.0002
            
            cart_pos = 5.
            pole_ang = 6.5
            pole_middle = -5.5
            cart_middle = -30.
            
            # vel_cart = 0.5
            
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        soft_torque_limit = 100.


    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            cart_pos = 2.
            cart_vel = 1.5
            pole_pos = 2.
            pole_vel = 1.5
            torque = 1.0  # force.
            episode = 1.0
        clip_observations = 100.
        clip_actions = 100.


    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            cart_pos = 0.3
            cart_vel = 0.3
            pole_pos = 0.3
            pole_vel = 0.5
        class static_noise:
            st_noise = True
            st_noise_cart_pos = 0.02
            st_noise_cart_vel = 0.05
            st_noise_pole_pos = 0.03
            st_noise_pole_vel = 0.05


    # viewer camera:
    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim(LeggedRobotCfg.sim):
        dt =  0.0011
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 4
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class cartpolesim2simCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 5.e-4
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'cartpole'

