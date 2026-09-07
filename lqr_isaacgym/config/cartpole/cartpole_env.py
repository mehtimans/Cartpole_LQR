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
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

from cartpole_sim2real import LEGGED_GYM_ROOT_DIR
from cartpole_sim2real.envs.base.legged_robot_config import LeggedRobotCfg
import isaacgym

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
from isaacgym.terrain_utils import *

import torch
from cartpole_sim2real.utils.helpers import class_to_dict
from cartpole_sim2real.envs.base.legged_robot import LeggedRobot
from cartpole_sim2real.utils.math import wrap_to_pi
import os
import numpy as np
from collections import deque
from isaacgym import gymapi
import time

class CARTPOLEFreeEnv(LeggedRobot):
    '''
    CARTPOLEFreeEnv is a class that represents a custom environment for a legged robot.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless): # Not logging errors, so I didn’t use the related variable.
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()
        
        if self.cfg.control.control_type == 'Actuator_network':
            self.load_actuator_net()

    def load_actuator_net(self):
        # Building path to the JIT actuator net file
        actuator_net_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'cartpole_sim2real', 'actuator_net')
        actuator_jit_path = os.path.join(actuator_net_dir, '2025-09-28_15-50-51_/JIT_model.pt')
        # Loading JIT actuator net
        self.actuator_net = torch.jit.load(actuator_jit_path).to(device=self.device)
        print("------ actuator net Loaded!!!! ------")
        print(f"Loading actuator jit model from: {actuator_jit_path}")
    
    def create_sim(self): # Modified to use only a plane, as we only need flat terrain.
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()

    def _init_buffers(self):    

        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state) # (0-3 position base, 3-7 quat base, 7-10 lin vel, 10-13 ang vel)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg).to(device=self.device)
        self.static_noise = self._get_static_noise_vec().to(device=self.device)
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_cart_vel = torch.zeros_like(self.dof_vel[:, 0])
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.random_dof = torch.FloatTensor(64).uniform_(-0.1, 0.1)  # Randomization for pole angle dof_pos[1]
        # define external force
        self.exforce = torch.zeros((self.num_envs, self.num_bodies, 3), device = self.device, dtype = torch.float)
        self.extorque = torch.zeros((self.num_envs, self.num_bodies, 3), device = self.device, dtype = torch.float)
        
        # kp and kd for control position control
        self.p_gains = torch.tensor(self.cfg.control.slider_to_cart_stiffness,
                            dtype=torch.float, device=self.device)
        self.d_gains = torch.tensor(self.cfg.control.slider_to_cart_damping,
                            dtype=torch.float, device=self.device)
        
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle

        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))
        
        # Initialize actuator network parameters
        self.error_history = deque(maxlen=self.cfg.control.actuator_frame_stack)
        self.vel_history = deque(maxlen=self.cfg.control.actuator_frame_stack)
        
        for _ in range(self.cfg.control.actuator_frame_stack):
            self.error_history.append(torch.zeros(
                self.num_envs, 1, dtype=torch.float, device=self.device))
            self.vel_history.append(torch.zeros(
                self.num_envs, 1, dtype=torch.float, device=self.device))
            
        self.actuator_net_input = np.zeros([self.num_envs, 6], dtype=np.float32)
        self.pulley_R = self.cfg.control.pulley_radius
        
        # Precompute a buffer of randomized DOF position offsets for cart and pole.
        self.num_buckets_dof = self.cfg.domain_rand.num_buckets_dof
        cart_dof_range = self.cfg.domain_rand.cart_dof_range
        pole_dof_range = self.cfg.domain_rand.pole_dof_range
        cart_dof_buckets = torch_rand_float(cart_dof_range[0], cart_dof_range[1], (self.num_buckets_dof,1), device=self.device)
        pole_dof_buckets = torch_rand_float(pole_dof_range[0], pole_dof_range[1], (self.num_buckets_dof,1), device=self.device)
        self.dof_pos_rand = torch.cat((cart_dof_buckets, pole_dof_buckets),dim=-1)
    
    def _push_force(self):
        "push external force ['slider', 'cart', 'pole'] body props [env,body,xyz]"
        # this method was modified and cleaned on May 13,2025 
        
        max_force = self.cfg.domain_rand.max_push_force
        self.rand_push_force = torch_rand_float(-max_force, max_force, (self.num_envs, 1), device=self.device)
        
        self.exforce [:,2,1] = self.rand_push_force.squeeze(1)
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.exforce), gymtorch.unwrap_tensor(self.extorque), gymapi.ENV_SPACE)
        
    def _init_privilaged(self):
        ## TODO: using for privileged observation
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.env_restitution = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.pole_joint_damping =  torch.zeros(self.num_envs, 1,dtype=torch.float, device=self.device, requires_grad=False)
        self.slider_joint_damping =  torch.zeros(self.num_envs, 1,dtype=torch.float, device=self.device, requires_grad=False)
        self.pole_joint_friction =  torch.zeros(self.num_envs, 1,dtype=torch.float, device=self.device, requires_grad=False)
        self.slider_joint_friction =  torch.zeros(self.num_envs, 1,dtype=torch.float, device=self.device, requires_grad=False)
        self.pole_joint_stiffness =  torch.zeros(self.num_envs, 1,dtype=torch.float, device=self.device, requires_grad=False)
        self.slider_joint_stiffness =  torch.zeros(self.num_envs, 1,dtype=torch.float, device=self.device, requires_grad=False)
        self.cart_payloads = torch.zeros(self.num_envs, 1,dtype=torch.float, device=self.device, requires_grad=False)
        self.pole_payloads = torch.zeros(self.num_envs, 1,dtype=torch.float, device=self.device, requires_grad=False)
        self.cart_com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.pole_com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.rand_push_force = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure
        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_obs)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:1] = noise_scales.cart_pos * noise_level * self.obs_scales.cart_pos     # cart dof pos
        noise_vec[1:2] = noise_scales.cart_vel * noise_level * self.obs_scales.cart_vel     # cart dof vel
        noise_vec[3:4] = noise_scales.pole_pos * noise_level * self.obs_scales.pole_pos     # pole dof pos
        noise_vec[4:5] = noise_scales.pole_vel * noise_level * self.obs_scales.pole_vel     # pole dof vel
        noise_vec[5:6] = 0                                                                  # previous actions
        
        return noise_vec
    
    def _get_static_noise_vec(self):
        self.static_noise = torch.zeros(self.num_envs, self.cfg.env.num_single_obs, 
                                               dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.noise.static_noise.st_noise:
            # Bounds per observation element
            bounds = torch.tensor([
                self.cfg.noise.static_noise.st_noise_cart_pos,
                self.cfg.noise.static_noise.st_noise_cart_vel,
                self.cfg.noise.static_noise.st_noise_pole_pos,
                self.cfg.noise.static_noise.st_noise_pole_vel,
                0.0  # No noise on the last element
            ], dtype=torch.float32, device=self.device)

            random_factors = (torch.rand((self.num_envs, self.cfg.env.num_single_obs), device=self.device) - 0.5) * 2
            self.static_noise[:, :4] = random_factors[:, :4] * bounds[:4]

            # print("static noise #################", self.static_noise) 

        return self.static_noise

    def _process_dof_props(self, props, env_id): 
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        # this method needs to be checked. joint damping is changed (mehtimans)
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        
        # set joint damping, joint damping is set individually for each dof, the slider has higher damping then the pole joint
        if self.cfg.domain_rand.randomize_joint_damping:
            slider_joint_damping_range = self.cfg.domain_rand.slider_joint_damping_range
            pole_joint_damping_range = self.cfg.domain_rand.pole_joint_damping_range
            self.pole_joint_damping[env_id, 0] = np.random.uniform(pole_joint_damping_range[0], pole_joint_damping_range[1])
            self.slider_joint_damping[env_id, 0] = np.random.uniform(slider_joint_damping_range[0], slider_joint_damping_range[1])
        else:
            self.slider_joint_damping[env_id, 0] = self.cfg.domain_rand.default_slider_joint_damping_range
            self.pole_joint_damping[env_id, 0] = self.cfg.domain_rand.default_pole_joint_damping_range
        
        # set joint friction, joint friction is set individually for each dof, the slider has higher friction then the pole joint
        if self.cfg.domain_rand.randomize_joint_friction:
            slider_joint_friction_range = self.cfg.domain_rand.slider_joint_friction_range
            pole_joint_friction_range = self.cfg.domain_rand.pole_joint_friction_range
            self.pole_joint_friction[env_id, 0] = np.random.uniform(pole_joint_friction_range[0], pole_joint_friction_range[1])
            self.slider_joint_friction[env_id, 0] = np.random.uniform(slider_joint_friction_range[0], slider_joint_friction_range[1])
        else:
            self.slider_joint_friction[env_id, 0] = self.cfg.domain_rand.default_slider_joint_friction_range
            self.pole_joint_friction[env_id, 0] = self.cfg.domain_rand.default_pole_joint_friction_range
        
        # set joint stiffness, both dofs receive the same value
        if self.cfg.domain_rand.randomize_joint_stiffness:
            slider_joint_stiffness_range = self.cfg.domain_rand.slider_joint_stiffness_range
            pole_joint_stiffness_range = self.cfg.domain_rand.pole_joint_stiffness_range
            self.pole_joint_stiffness[env_id, 0] = np.random.uniform(pole_joint_stiffness_range[0], pole_joint_stiffness_range[1])
            self.slider_joint_stiffness[env_id, 0] = np.random.uniform(slider_joint_stiffness_range[0], slider_joint_stiffness_range[1])
        else:
            self.slider_joint_stiffness[env_id, 0] = self.cfg.domain_rand.default_slider_joint_stiffness_range
            self.pole_joint_stiffness[env_id, 0] = self.cfg.domain_rand.default_pole_joint_stiffness_range
            
        props['damping'][0] = self.slider_joint_damping[env_id, 0] # slider to cart 
        props['damping'][1] = self.pole_joint_damping[env_id, 0] # cart to pole 
        props['friction'][0] = self.slider_joint_friction[env_id, 0] # slider to cart 
        props['friction'][1] = self.pole_joint_friction[env_id, 0] # cart to pole 
        props['stiffness'][0] = self.slider_joint_stiffness[env_id, 0] # slider to cart 
        props['stiffness'][1] = self.pole_joint_stiffness[env_id, 0] # cart to pole 
        
        return props


    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        # this method was checked and cleaned on may 13,2025 (mehtimans)
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                num_buckets_fri = self.cfg.domain_rand.num_buckets_friction
                bucket_ids = torch.randint(0, num_buckets_fri, (self.num_envs, 1))

                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets_fri,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
            
            self.env_frictions[env_id] = self.friction_coeffs[env_id]
        else:
            for s in range(len(props)):
                props[s].friction = self.cfg.domain_rand.friction_default
            self.env_frictions[env_id] = self.cfg.domain_rand.friction_default

        
        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                num_buckets_res = self.cfg.domain_rand.num_buckets_restitution
                bucket_ids = torch.randint(0, num_buckets_res, (self.num_envs, 1))

                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets_res,1), device='cpu')
                self.restitution_coeffs = restitution_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
            
            self.env_restitution[env_id] = self.restitution_coeffs[env_id]
        else:
            for s in range(len(props)):
                props[s].restitution = self.cfg.domain_rand.restitution_default
            
            self.env_restitution[env_id] = self.cfg.domain_rand.restitution_default
        
        return props

    def _process_rigid_body_props(self, props, env_id): 
        # this method was checked and cleaned on may 13,2025 (mehtimans)
          
        # add extra mass to the bodies    
        if self.cfg.domain_rand.randomize_base_mass:       
            cart_mass_range = self.cfg.domain_rand.added_cart_mass_range
            pole_mass_range = self.cfg.domain_rand.added_pole_mass_range
            self.cart_payloads[env_id, 0] = np.random.uniform(cart_mass_range[0], cart_mass_range[1])   
            self.pole_payloads[env_id, 0] = np.random.uniform(pole_mass_range[0], pole_mass_range[1])   
            
            props[1].mass += self.cart_payloads[env_id, 0]  # add mass to cart
            props[2].mass += self.pole_payloads[env_id, 0]  #add mass to pole
        # set a center of mass offset for each body
        if self.cfg.domain_rand.randomize_com_displacement:
            cart_com_xrange = self.cfg.domain_rand.cart_com_displacement_xrange
            cart_com_yrange = self.cfg.domain_rand.cart_com_displacement_yrange
            cart_com_zrange = self.cfg.domain_rand.cart_com_displacement_zrange
            
            pole_com_xrange = self.cfg.domain_rand.pole_com_displacement_xrange
            pole_com_yrange = self.cfg.domain_rand.pole_com_displacement_yrange
            pole_com_zrange = self.cfg.domain_rand.pole_com_displacement_zrange
            
            self.cart_com_displacements[env_id, :] = torch.cat((
                                    torch.tensor([np.random.uniform(cart_com_xrange[0], cart_com_xrange[1])],dtype=torch.float32),  
                                    torch.tensor([np.random.uniform(cart_com_yrange[0], cart_com_yrange[1])],dtype=torch.float32),   
                                    torch.tensor([np.random.uniform(cart_com_zrange[0], cart_com_zrange[1])],dtype=torch.float32)),dim=-1)
            
            self.pole_com_displacements[env_id, :] = torch.cat((
                                    torch.tensor([np.random.uniform(pole_com_xrange[0], pole_com_xrange[1])]),  
                                    torch.tensor([np.random.uniform(pole_com_yrange[0], pole_com_yrange[1])]),   
                                    torch.tensor([np.random.uniform(pole_com_zrange[0], pole_com_zrange[1])]) ),dim=-1)  

            props[1].com += gymapi.Vec3(self.cart_com_displacements[env_id, 0], self.cart_com_displacements[env_id, 1], #cart
                                        self.cart_com_displacements[env_id, 2])
            props[2].com += gymapi.Vec3(self.pole_com_displacements[env_id, 0], self.pole_com_displacements[env_id, 1], #pole
                                        self.pole_com_displacements[env_id, 2])                                      
        return props
    
    def pre_physics_step(self, actions):
        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
        actions = actions.to(device=self.device)
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions
        return actions
    
    def step(self, actions):
        st_time = time.time()
        # this method was modified and cleaned on May 31,2025 (mehtimans)
        actions = self.pre_physics_step(actions)
        clip_actions = self.cfg.normalization.clip_actions 
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            pole_torque = torch.zeros_like(self.torques)        # We don’t apply torque to the pole, so we just set it to zero.
            torques = torch.cat((self.torques, pole_torque), dim=-1)

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        # print('torque:',torques)
        # print('------------------------------------------------------------------')
        
        # print('vel cart:',self.dof_vel[:, 0])
        self.post_physics_step()
        
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def post_physics_step(self): 
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        # self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_cart_vel[:] = self.dof_vel[:, 0]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def compute_actuator_net_input(self, x_des):
        """ Build input features for actuator network from error/velocity history. """
        self.error_history.append(x_des + self.default_dof_pos[:, 0] - self.dof_pos[:, 0:1])
        self.vel_history.append(self.dof_vel[:, 0:1])

        e0  = self.error_history[self.cfg.control.actuator_frame_stack - 1]          # current
        v0  = self.vel_history[self.cfg.control.actuator_frame_stack - 1]
        e10 = self.error_history[self.cfg.control.actuator_frame_stack - 1 - 10]     # 10 steps ago
        v10 = self.vel_history[self.cfg.control.actuator_frame_stack - 1 - 10]
        e20 = self.error_history[self.cfg.control.actuator_frame_stack - 1 - 20]     # 20 steps ago
        v20 = self.vel_history[self.cfg.control.actuator_frame_stack - 1 - 20]

        self.actuator_net_input = torch.cat([e0, v0, e10, v10, e20, v20], dim=1)

    def compute_observations(self):     # These functions were called at the beginning but didn’t affect the outcome, so they were removed: 
                                        # _get_phase, compute_ref_state, compute_contact_states.
        """ Compute observations for the cartpole. """
        self.privileged_obs_buf = torch.cat(( 
                                              self.dof_pos[:, 0].unsqueeze(1) * self.obs_scales.cart_pos ,
                                              self.dof_vel[:, 0].unsqueeze(1) * self.obs_scales.cart_vel,
                                              self.dof_pos[:, 1].unsqueeze(1) * self.obs_scales.pole_pos ,
                                              self.dof_vel[:, 1].unsqueeze(1) * self.obs_scales.cart_vel ,
                                              self.actions,
                                            #   self.rand_push_force,
                                            #   self.pole_com_displacements[:, 2].unsqueeze(1),
                                            #   self.slider_joint_friction,
                                            #   self.pole_joint_friction,
                                            #   self.cart_payloads,
                                            #   self.pole_payloads,
                                            #   self.torques* self.obs_scales.torque,
                                            #   self.episode_length_buf.unsqueeze(1)* self.obs_scales.episode,
                                              ),dim=-1)
                                         
        obs_buf = torch.cat((
                                self.dof_pos[:, 0].unsqueeze(1) * self.obs_scales.cart_pos ,
                                self.dof_vel[:, 0].unsqueeze(1) * self.obs_scales.cart_vel,
                                self.dof_pos[:, 1].unsqueeze(1) * self.obs_scales.pole_pos ,
                                self.dof_vel[:, 1].unsqueeze(1) * self.obs_scales.cart_vel ,
                                self.actions
                                ),dim=-1)
        
        if self.add_noise:  
            dynamic_noise = ((torch.rand_like(obs_buf)-0.5) * 2) * self.noise_scale_vec * obs_buf.abs()
            noise = dynamic_noise + self.static_noise
            obs_buf += noise
            
        self.obs_history.append(obs_buf.clone())
        self.critic_history.append(self.privileged_obs_buf.clone())
        
        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

        # print('obs buf:', obs_buf)
        # print('privileged obs:', self.privileged_obs_buf)
        # print('-------------------------------------------------------------------------------')

    def reset_idx(self, env_ids): 
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids) and Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_cart_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
            
    def _reset_dofs(self,env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within random bucket.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if self.cfg.domain_rand.randomize_dof_pos:
            dof_rand_indices = torch.randint(0, self.num_buckets_dof, (len(env_ids),), device=self.device)
            dof_rand_offsets = self.dof_pos_rand[dof_rand_indices] 
            
            self.dof_pos[env_ids] = self.default_dof_pos.unsqueeze(0) + dof_rand_offsets
        else:
            self.dof_pos[env_ids] = self.default_dof_pos.unsqueeze(0)
        
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos[:, 0] - self.dof_pos[:, 0:1]) - self.d_gains * self.dof_vel[:, 0:1]
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        elif control_type=="Actuator_network":
            self.compute_actuator_net_input(actions_scaled)
            torques = self.actuator_net(self.actuator_net_input)
            torques = torques / self.pulley_R
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits[1], self.torque_limits[1])   # Used self.torque_limits[1] instead of self.torque_limits to avoid a dimension error, 
                                                                                    # since only one actuator is used (not two).

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

    def _create_envs(self): 
                            
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """

        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
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

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset) #['slider', 'cart', 'pole'] body props
        self.dof_names = self.gym.get_asset_dof_names(robot_asset) #['slider_to_cart', 'cart_to_pole']
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        
        self._init_privilaged()

        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pos = gymapi.Transform()
        start_pos.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        
        self.actor_handles = []
        self.envs = []

        self.joint_type = []
        for i in range(self.num_dofs + 1):
             self.joint_type.append(self.gym.get_asset_joint_type(robot_asset, i))

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pos.p = gymapi.Vec3(*pos)
            
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pos, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        
    def _parse_cfg(self, cfg):  # Removed command_ranges and set self.cfg.terrain.curriculum = False, 
                                # deleting the related condition since terrain is always a plane in cartpole.
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        
        # Calculating the number of steps for applying external force
        self.push_force_interval = np.ceil(self.cfg.domain_rand.push_force_interval_s / self.dt)

    def check_termination(self): # Modified the function to reset if the pole's angle exceeds pi/2.
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1) # Pole
        pole_angle_check = abs(self.dof_pos[:, 1].to(self.device).clone().detach()) > np.pi / 2
        # cart_pos_check = abs(self.dof_pos[:, 0].to(self.device).clone().detach()) > self.cfg.env.reset_dist
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= pole_angle_check
        # self.reset_buf |= cart_pos_check

    

    def compute_reward(self): # Implemented reward function from walking_these_way repo and added a new way to compute rewards.
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            rew = rew.to(self.device) 
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    
    def _post_physics_step_callback(self): # Removed resample_command, heading_command, and measure_heights as they are not used in this setup.
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        if self.cfg.domain_rand.push_force and  (self.common_step_counter % self.push_force_interval == 0):
            self._push_force()
    

    # ================================================ Rewards ================================================== #
    def _reward_remain_up(self):
        reward = torch.cos(self.dof_pos[:, 1].to(self.device).clone().detach()) - 1.0 * torch.square(1.5 *(self.dof_vel[:, 0]))
        reward = torch.where(torch.abs(self.dof_pos[:, 0].to(self.device).clone().detach()) > 0.15, torch.ones_like(reward) * -2.0, reward)
        return reward
    
    def _reward_vel_tracking(self):
        return 1.0 - 0.5 * self.dof_pos[:, 1] ** 2 - 0.1 * self.dof_pos[:, 0] ** 2 - 0.01 * self.dof_vel[:, 0] ** 2 - 0.01 * self.dof_vel[:, 1] ** 2 
    
    def _reward_decay(self): # cart position penalty
        return torch.exp((-2.0 * torch.abs(self.dof_pos[:, 0])) * torch.exp( -1 * torch.abs(self.dof_pos[:, 1])))
    
    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_balance(self):
        reward=torch.where((torch.abs(self.dof_pos[:, 0]) < 0.05 ) & (torch.abs(self.dof_pos[:, 1]) < 0.1), 1 , 0 )
        # print('++++++++++++++++++++++++++++', reward)
        return reward
    
    def _reward_torque_limits(self):
    # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _reward_cart_acc(self):
        # Penalize dof accelerations
        # print("shape",np.shape(self.last_dof_vel[:, 0] ),np.shape(self.dof_vel[:, 0] ))
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_terminations(self):
    # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_cart_pos(self):
        reward = 1.0 - self.dof_pos[:, 1] ** 2 - 0.01 * torch.abs(self.dof_vel[:, 0]) - 0.005 * torch.abs(self.dof_vel[:, 1])
        return torch.where(torch.abs(self.dof_pos[:, 0]) > self.cfg.env.reset_dist, torch.ones_like(reward) * -2.0, reward )
    
    def _reward_pole_ang(self):
        reward = 1.0 - self.dof_pos[:, 1] ** 2 - 0.01 * torch.abs(self.dof_vel[:, 0]) - 0.005 * torch.abs(self.dof_vel[:, 1])
        return torch.where(torch.abs(self.dof_pos[:, 1]) > np.pi / 4 , torch.ones_like(reward) * -2.0, reward )
    
    
    
    def _reward_pole_middle(self):
        penalty= torch.abs(self.dof_pos[: ,1])
        return penalty
    
    def _reward_cart_middle(self):
        penalty= torch.abs(self.dof_pos[: , 0])
        return penalty

    # def _reward_vel_cart(self):
    #     return -0.05 * torch.abs(self.dof_vel[:,0])



    
    
