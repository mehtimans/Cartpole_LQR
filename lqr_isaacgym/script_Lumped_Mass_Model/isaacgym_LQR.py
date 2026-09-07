import os
import numpy as np
import sys 

import pandas as pd

from collections import deque

from scipy.spatial.transform import Rotation as R
from cartpole_sim2real import LEGGED_GYM_ROOT_DIR
from cartpole_sim2real.envs import Cartpolesim2simCfg
from cartpole_sim2real.utils.helpers import class_to_dict, parse_sim_params
from cartpole_sim2real.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

from isaacgym import gymapi
from isaacgym import gymutil

import matplotlib.pyplot as plt
import torch
import time
from datetime import datetime
import math
import scipy.linalg

class isaacgym_test():
    def __init__(self):
        self.cfg = Cartpolesim2simCfg()
        self.gym = gymapi.acquire_gym()
        self.physics_engine = PHYSICS_ENGINE
        self.sim_device = SIM_DEVICE
        self.device = DEVICE
        self.log_itr = int(LOG_ITR)
        self.sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.graphics_device_id = self.sim_device_id
        self.sim_parameters = self.sim_params()
        self.create_sim() 
        self.gym.prepare_sim(self.sim)
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.set_camera()

        
        self.num_envs = 1
        self.num_actions = 1
        self.num_states = 4

        self._init_buffers()

        # noise
        self.add_noise = False
        self.noise_level = 1 # scales other values
        self.noise_cart_pos = 0.3
        self.noise_cart_vel = 0.3
        self.noise_pole_pos = 0.3
        self.noise_pole_vel = 0.3

        self.st_noise_cart_pos = 0.1
        self.st_noise_cart_vel = 0.2
        self.st_noise_pole_pos = 0.5
        self.st_noise_pole_vel = 1

        self._get_noise_scale_vec()

        #LQR
        self.ref = torch.tensor([0, 0, np.pi, 0], dtype=torch.float, device=self.device) # Desired state

    def lqr(self, A, B, Q, R):
        """
            Solve the continuous time lqr controller.     
            dx/dt = A x + B u     
            cost = integral x.T*Q*x + u.T*R*u
        """
        #ref Bertsekas, p.151 
        #first, try to solve the ricatti equation
        X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))     
        #compute the LQR gain
        K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
        eigVals, eigVecs = scipy.linalg.eig(A-B*K)     
        return K, X, eigVals

    def compute_lqr_gain(self): 
        
        m = 0.175       # Pendulum mass [kg] pole
        M = 0.694       # cart
        L = 0.287       # Pendulum length [m]
        g = 9.8065      # Gravitational acceleration [m/s^2]
        d = 0.7         # Friction coefficient

        A = np.array([[0, 1, 0, 0],
              [0, -d/M, -m*g/M, 0], 
              [0, 0, 0, 1], 
              [0, -d/(M*L), -(m+M)*g/(M*L), 0]])

        B = np.array([[0],
                      [1/M],
                      [0],
                      [1/(M*L)]])
        
        C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

        Q = np.diag([100, 30,  5000, 25]) 
        R = np.array([[0.0005]])

        self.K, _, eigVals = self.lqr(A, B, Q, R)
        self.K = torch.tensor(self.K, dtype=torch.float, device=self.device)
        

        print('k:', self.K)
        print('eigVals', eigVals)
        print('A:', A)
        print('B:', B)
        print('Q:', Q)
        print('R:', R)
        
        print('----------------------------------------------------------------------')
        # time.sleep(10000)
        
    def run(self):

        self.compute_lqr_gain()
        self.states = self.compute_states().to(device=self.device)
        # print('first state = ',self.states)
        
        try:
            for step in range (int(MAX_ITR)):
                self.gym.fetch_results(self.sim, True)  
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # if step%30==0:
                #     self._push_force(force=0)
                #     print('********push force!!!!step%30**********')
                # elif step%70==0:
                #     self._push_force(force=0)
                #     print('********push force!!!!step%70**********')
                
                self.u = -self.K @ (self.states - self.ref).T 
                # self.u = torch.tensor(self.u, dtype=torch.float, device=self.device, requires_grad=False)
                self.u = self.u.clone().detach().to(dtype=torch.float, device=self.device)
                torque2 = torch.zeros_like(self.u)                # We don’t apply torque to the pole, so we just set it to zero.
                torques = torch.cat((self.u, torque2), dim=-1) 
                self.act(torques)
        
                self.states = self.compute_states().to(device=self.device)
                
                if self.gym.query_viewer_has_closed(self.viewer):
                    break

                self.state_log.append(self.states[0].cpu().numpy().copy())
                self.control_log.append(self.u[0].cpu().numpy().copy())
                self.time_log.append(step * 0.005)

        except KeyboardInterrupt:
                print("\nRun interrupted by user.")

        finally:
                self.save_log() 
                self.plot_log() 
                print("Logs saved and plot generated.")        

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def compute_states(self): 

        self.gym.refresh_actor_root_state_tensor(self.sim)
   
        self.states = torch.cat((   
                                self.dof_pos[:, 0].unsqueeze(1) ,
                                self.dof_vel[:, 0].unsqueeze(1) ,
                                self.dof_pos[:, 1].unsqueeze(1) + np.pi,
                                self.dof_vel[:, 1].unsqueeze(1) ,
                                ),dim=-1)

        if self.add_noise:  
            dynamic_noise = ((torch.rand_like(self.states)-0.5) * 2) * self.noise_vec * self.states.abs()
            self.static_noise = torch.cat((
                                    torch.tensor([np.random.uniform(-self.st_noise_cart_pos, self.st_noise_cart_pos)],dtype=torch.float32),  
                                    torch.tensor([np.random.uniform(-self.st_noise_cart_vel, self.st_noise_cart_vel)],dtype=torch.float32),   
                                    torch.tensor([np.random.uniform(-self.st_noise_pole_pos, self.st_noise_pole_pos)],dtype=torch.float32),   
                                    torch.tensor([np.random.uniform(-self.st_noise_pole_vel, self.st_noise_pole_vel)],dtype=torch.float32)),dim=-1).to(device=self.device)
            
            noise = dynamic_noise + self.static_noise
            self.states += noise
        
        return self.states
    
    def act(self, torques):
        # print("torques ", torques)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
        self.gym.simulate(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
    

    def _get_noise_scale_vec(self):
        
        self.noise_vec = torch.zeros(self.num_states).to(device=self.device)
        self.noise_vec[0:1] = self.noise_cart_pos * self.noise_level    # cart dof pos
        self.noise_vec[1:2] = self.noise_cart_vel * self.noise_level    # cart dof vel
        self.noise_vec[3:4] = self.noise_pole_pos * self.noise_level    # pole dof pos
        self.noise_vec[4:5] = self.noise_pole_vel * self.noise_level    # pole dof vel



    def _push_force(self, force):
        "push external force ['slider', 'cart', 'pole'] body props [env,body,xyz]"
        # this method was modified and cleaned on May 13,2025 
        
        max_force = self.cfg.domain_rand.max_push_force
        self.rand_push_force = torch_rand_float(-max_force, max_force, (self.num_envs, 1), device=self.device)
        
        # self.exforce [:,2,1] = self.rand_push_force.squeeze(1)
        self.exforce [:,2,1] = force
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.exforce), gymtorch.unwrap_tensor(self.extorque), gymapi.ENV_SPACE)
    
    def _init_buffers(self):
        
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.default_dof_pos = torch.zeros(1, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        self.static_noise = torch.zeros(self.num_envs, self.num_states, dtype=torch.float, device=self.device, requires_grad=False)
        self.states = torch.zeros(self.num_envs, self.num_states, dtype=torch.float, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.torqus = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.exforce = torch.zeros((self.num_envs, self.num_bodies, 3), device = self.device, dtype = torch.float)
        self.extorque = torch.zeros((self.num_envs, self.num_bodies, 3), device = self.device, dtype = torch.float)

        self.state_log = []
        self.control_log = []
        self.time_log = []

    def sim_params(self):

        sim_params = gymapi.SimParams()
        sim_params_config = {"sim": class_to_dict(self.cfg.sim)}
        gymutil.parse_sim_config(sim_params_config["sim"], sim_params)
        args = gymutil.parse_arguments(description="View default args")

        sim_params.physx.use_gpu = args.use_gpu 
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        sim_params.physx.num_threads = args.num_threads
        sim_params.dt = 0.002  # or your desired timestep
        sim_params.substeps = 1  # more substeps = better stability

        # for attr in dir(sim_params):
        #     if not attr.startswith('_') and not callable(getattr(sim_params, attr)):
        #         value = getattr(sim_params, attr)
        #         print(f"{attr}: {value}")

        # for attr in dir(sim_params.physx):
        #     if not attr.startswith('_') and not callable(getattr(sim_params.physx, attr)):
        #         print(f"physx.{attr}: {getattr(sim_params.physx, attr)}")
    
        # print("############################### args", vars(args))

        return sim_params
    
    def create_sim(self): 
        """ Creates simulation, flat plain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_parameters)
        self._create_ground_plane()
        self._create_envs()

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

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
        
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pos = gymapi.Transform()
        start_pos.p = gymapi.Vec3(*self.base_init_state[:3])

        self.joint_type = []
        for i in range(self.num_dofs + 1):
             self.joint_type.append(self.gym.get_asset_joint_type(robot_asset, i))

        # Create single environment
        env_handle = self.gym.create_env(self.sim, gymapi.Vec3(), gymapi.Vec3(), 1)

        # Create actor
        actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pos, self.cfg.asset.name, 0, self.cfg.asset.self_collisions, 0)

        dof_states = self.gym.get_actor_dof_states(env_handle, actor_handle, gymapi.STATE_ALL)
        dof_states['pos'][0] = 0 # Assuming index 1 is the cart-to-pole joint
        dof_states['pos'][1] = -np.pi/8 # Assuming index 1 is the cart-to-pole joint
        self.gym.set_actor_dof_states(env_handle, actor_handle, dof_states, gymapi.STATE_ALL)

        # Set DOF properties
        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        dof_props['stiffness'][:] = 0.3
        dof_props['damping'][0] = 0.7   # slider to cart
        dof_props['damping'][1] = 0.001 # cart to pole
        dof_props['friction'][0] = 0.3  # slider to cart 
        dof_props['friction'][1] = 0.0 # cart to pole
        self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

        # body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
        # body_props[1].mass += 0.05  # add mass to cart
        # body_props[2].mass += 0.01  # add mass to pole
        # cart_com_displacements = torch.tensor([0, 0.0, 0.02], dtype=torch.float, device=self.device, requires_grad=False)
        # pole_com_displacements = torch.tensor([0, 0.0, 0.02], dtype=torch.float, device=self.device, requires_grad=False)
        # body_props[1].com += gymapi.Vec3(cart_com_displacements[0], cart_com_displacements[1], #cart
        #                                  cart_com_displacements[2])
        # body_props[2].com += gymapi.Vec3(pole_com_displacements[0], pole_com_displacements[1], #pole
        #                                  pole_com_displacements[2])
        # self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
      
        # Save handles
        self.envs = [env_handle]
        self.actor_handles = [actor_handle]

    def set_camera(self):
        """ Set camera position and direction
        """
        position = [-2, -5, 4]  # [m]
        lookat = [11., 20, -10.]  # [m]

        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def save_log(self):

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "log")
        self.log_dir = os.path.join(log_root, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)

        log_df = pd.DataFrame({
            'time': self.time_log,
            'x': [s[0] for s in self.state_log],
            'x_dot': [s[1] for s in self.state_log],
            'theta': [s[2] for s in self.state_log],
            'theta_dot': [s[3] for s in self.state_log],
            'torque': [u[0] for u in self.control_log],
        })

        self.log_path = os.path.join(self.log_dir, "lqr_log.csv")
        log_df.to_csv(self.log_path, index=False)
        print(f"Log saved to {self.log_path}")

    def plot_log(self):

        if not hasattr(self, 'log_path'):
            print("No log file to plot.")
            return

        log_df = pd.read_csv(self.log_path)

        for column in ['x', 'x_dot', 'theta', 'theta_dot', 'torque']:
            plt.figure(figsize=(8, 4))
            plt.plot(log_df['time'], log_df[column])
            plt.xlabel("Time (s)")
            plt.ylabel(column)
            plt.title(f"{column} over time")
            plt.grid(True)
            plt.tight_layout()

            # Save individual figure
            fig_path = os.path.join(self.log_dir, f"{column}.png")
            plt.savefig(fig_path)
            print(f"Saved: {fig_path}")

if __name__ == '__main__':
   
    SELF_BALANCE_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    DEVICE = 'cuda'
    PHYSICS_ENGINE = gymapi.SIM_PHYSX
    SIM_DEVICE = 'cuda'
    MAX_ITR = 1e5
    LOG_ITR = 2e3

    model = isaacgym_test()
    model.run()
