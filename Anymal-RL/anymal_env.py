import torch
import math
from pathlib import Path
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import numpy as np

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class ObsDict(dict):
    def to(self, device):
        return ObsDict({k: v.to(device) if hasattr(v, 'to') else v for k, v in self.items()})

class QuadV3:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = {"policy": obs_cfg["num_obs"]}
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.cfg = env_cfg

        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        joint_names = self.env_cfg["joint_names"]
        default_angles = self.env_cfg["default_joint_angles"]
        max_torques = self.env_cfg["max_torques"]
        if len(joint_names) != self.num_actions:
            raise ValueError("joint_names must contain exactly num_actions entries")
        if set(joint_names) != set(default_angles):
            raise ValueError("default_joint_angles must contain exactly the configured joint_names")
        if len(max_torques) != self.num_actions:
            raise ValueError("max_torques must contain one limit per action")
        self.terrain_spawn = 0.0

        # Scene Setup
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=self.env_cfg.get("enable_self_collision", True),
                tolerance=1e-5,
                max_collision_pairs=40,
                batch_dofs_info=True,
                batch_links_info=True,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(self.terrain_spawn + 2.5, self.terrain_spawn + 2.5, 2.5), # Offset from spawn
                camera_lookat=(self.terrain_spawn, self.terrain_spawn, 0.5),          # Look at spawn
                camera_fov=40, 
                max_FPS=int(1.0 / self.dt)
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

        

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # Robot Setup
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        urdf_path = Path(self.env_cfg["urdf_path"])
        if not urdf_path.is_absolute():
            urdf_path = Path(__file__).resolve().parent / urdf_path
        
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(urdf_path),
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )

        # Genesis merges the URDF's fixed imu_link into its parent base link.

        imu_link = self.robot.get_link(self.env_cfg["imu_link_name"])

        self.imu = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx=self.robot.idx,
                link_idx_local=imu_link.idx_local,
                pos_offset=self.env_cfg.get("imu_pos_offset", (0.0, 0.0, 0.0)),
                acc_noise=(0.0, 0.0, 0.0),
                gyro_noise=(0.0, 0.0, 0.0),
                interpolate=True,
                draw_debug=False,
            )
        )


        self.motors_dof_idx = [self.robot.get_joint(name).dofs_idx_local[0] for name in self.env_cfg["joint_names"]]

        self.scene.build(n_envs=num_envs)
        
        # 3. Apply all physics parameters AFTER the build
        # Buffer & Index Setup
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # --- EXPLICIT TORQUE LIMIT OVERRIDE ---
        # Fetch the list of per-joint URDF effort limits.
        max_torques = self.env_cfg.get("max_torques", [80.0] * self.num_actions)
        
        # negative and positive boundary lists
        lower_bounds = [-t for t in max_torques]
        upper_bounds = [t for t in max_torques]
        
        # Apply the specific limits to Genesis
        self.robot.set_dofs_force_range(lower_bounds, upper_bounds, self.motors_dof_idx)
        # --------------------------------------

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self._init_buffers()

    def _init_buffers(self):
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device) # IMU gyroscope
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device) # IMU orientation proxy
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs["policy"]), device=gs.device)        
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device)
        self.commands_scale = torch.tensor([self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]], device=gs.device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device)
        self.sampled_commands = torch.zeros_like(self.commands)

        # track exact torques
        self.torques = torch.zeros_like(self.actions)
        
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device)
        self.default_dof_pos = torch.tensor([self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]], device=gs.device)
        self.extras = {"observations": {}}

    def _resample_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return

        command_ranges = (
            self.command_cfg["lin_vel_x_range"],
            self.command_cfg["lin_vel_y_range"],
            self.command_cfg["ang_vel_range"],
        )
        for command_idx, command_range in enumerate(command_ranges):
            self.sampled_commands[envs_idx, command_idx] = gs_rand_float(
                command_range[0],
                command_range[1],
                (len(envs_idx),),
                gs.device,
            )

        # Pure forward/backward samples teach precise teleoperation speeds;
        # otherwise lateral and yaw commands are almost always present too.
        straight_commands = torch.rand(len(envs_idx), device=gs.device) < self.env_cfg.get(
            "straight_command_probability", 0.40
        )
        self.sampled_commands[envs_idx[straight_commands], 1:] = 0.0

        zero_commands = torch.rand(len(envs_idx), device=gs.device) < self.env_cfg.get(
            "zero_command_probability", 0.15
        )
        self.sampled_commands[envs_idx[zero_commands]] = 0.0

    def step(self, actions): # this is the step function that will be called by the RL loop, it takes in actions and returns obs, rewards, done, extras
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        target_dof_pos = self.actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)

        self.scene.step()

        # Extract the exact torque applied by the physics engine
        self.torques[:] = self.robot.get_dofs_control_force(self.motors_dof_idx)

        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True)
        
        # Ground-truth linear velocity (FOR REWARD ONLY)
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)

        # IMU reading (ONLY ONE CALL)
        imu_data = self.imu.read()

        self.base_ang_vel[:] = imu_data.ang_vel

        # Gravity projection using IMU orientation
        inv_base_quat = inv_quat(self.base_quat)

        self.projected_gravity = transform_by_quat(
            self.global_gravity,
            inv_base_quat
        )

        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # Resample first so the new command appears in the next observation.
        envs_idx = ((self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0).nonzero(as_tuple=False).reshape((-1,)))
        self._resample_commands(envs_idx)
        self.commands[:] = self.sampled_commands

        # Compute rewards on the terminal state before resetting timed-out
        # environments. The previous implementation rewarded the reset state.
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        timed_out = self.episode_length_buf >= self.max_episode_length
        unstable = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_height_lower_than"])
        )
        self.reset_buf = timed_out | unstable
        self.extras["time_outs"] = timed_out.float()
        self.extras.pop("episode", None)
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        self.obs_buf = torch.cat([
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
        ], axis=-1)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.extras["observations"]["critic"] = self.obs_buf
        obs_dict = ObsDict({"policy": self.obs_buf})

        # CAMERA TRACKING 
        if self.scene.viewer is not None:
            # robot coordinates
            robot_xyz = self.base_pos[0].detach().cpu().numpy()
            
            # Define where the camera is, where it looks, and which way is 'up'
            cam_pos = np.array([robot_xyz[0] + 2.5, robot_xyz[1] + 2.5, robot_xyz[2] + 1.5])
            cam_lookat = np.array([robot_xyz[0], robot_xyz[1], robot_xyz[2]])
            cam_up = np.array([0.0, 0.0, 1.0]) # In Genesis, Z is 'up'
            
            # Calculate camera axis vectors
            forward = cam_lookat - cam_pos
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(forward, cam_up)
            right = right / np.linalg.norm(right)
            
            true_up = np.cross(right, forward)
            
            # 4. Build the 4x4 Camera-to-World transformation matrix
            cam_pose = np.eye(4)
            cam_pose[:3, 0] = right      # X-axis (Right)
            cam_pose[:3, 1] = true_up    # Y-axis (Up)
            cam_pose[:3, 2] = -forward   # Z-axis (Camera looks down negative Z)
            cam_pose[:3, 3] = cam_pos    # Position
            
            #Apply it to the viewer
            self.scene.viewer.set_camera_pose(cam_pose)
        # If privileged obs are needed later, they go in extras
        # self.extras["privileged_obs"] = None 
        return obs_dict, self.rew_buf, self.reset_buf, self.extras
     
    def get_observations(self):
        return ObsDict({"policy": self.obs_buf})

    def get_privileged_observations(self):
 
        return None

    def reset_idx(self, envs_idx): 
        if len(envs_idx) == 0: return

        completed_envs = envs_idx[self.episode_length_buf[envs_idx] > 0]
        if len(completed_envs) > 0:
            self.extras["episode"] = {
                f"rew_{key}": self.episode_sums[key][completed_envs].mean().unsqueeze(0)
                for key in self.episode_sums
            }

        # Reset DOFs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(self.dof_pos[envs_idx], self.motors_dof_idx, zero_velocity=True, envs_idx=envs_idx)

        # Every episode starts in the nominal upright pose.
        new_pos = self.base_init_pos.repeat(len(envs_idx), 1)
        new_pos[:, 0] = self.terrain_spawn + gs_rand_float(-1.0, 1.0, (len(envs_idx),), gs.device)
        new_pos[:, 1] = self.terrain_spawn + gs_rand_float(-1.0, 1.0, (len(envs_idx),), gs.device)
        new_quats = self.base_init_quat.repeat(len(envs_idx), 1)

        self.base_pos[envs_idx] = new_pos
        self.base_quat[envs_idx] = new_quats
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.base_lin_vel[envs_idx] = 0.0
        self.torques[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        for key in self.episode_sums.keys(): self.episode_sums[key][envs_idx] = 0.0
        self._resample_commands(envs_idx)

        init_ang_vel = torch.zeros((len(envs_idx), 3), device=gs.device)
        inv_base_quat = inv_quat(self.base_quat[envs_idx])
        init_projected_gravity = transform_by_quat(
            self.global_gravity[envs_idx],
            inv_base_quat
        )
        self.base_ang_vel[envs_idx] = init_ang_vel
        self.projected_gravity[envs_idx] = init_projected_gravity
        self.commands[envs_idx] = self.sampled_commands[envs_idx]

    def reset(self): 
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))

        self.obs_buf = torch.cat([
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
        ], axis=-1)

        return ObsDict({"policy": self.obs_buf}), self.extras

    # --- REWARD FUNCTIONS ---
    def _reward_tracking_lin_vel(self):
        error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),
            dim=1,
        )
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])
        
    def _reward_tracking_ang_vel(self):
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])
        
    def _reward_base_height(self):
        error = (self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
        return torch.square(error)

    def _reward_lin_vel_z(self): 
        return torch.square(self.base_lin_vel[:, 2])
        
    def _reward_ang_vel_xy(self): 
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self): 
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        
    def _reward_similar_to_default(self): 
        error = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        return error

    def _reward_action_rate(self): 
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_torques(self): 
        return torch.sum(torch.square(self.torques), dim=1)
        
    def _reward_dof_vel(self): 
        return torch.sum(torch.square(self.dof_vel), dim=1)
        
    def _reward_dof_acc(self): 
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
