import math
from pathlib import Path
import torch
from tensordict import TensorDict
import genesis as gs

URDF_PATH = Path(__file__).resolve().parent / "cartpole1.urdf"

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class CartPoleEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device
        self.cfg = env_cfg

        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"].copy()

        # Create the Genesis scene.
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
                gravity=(0.0, 0.0, -9.81),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, -5.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(num_envs))),
            show_viewer=show_viewer,
        )

        # fixed=True keeps the slide bar fixed in the world.
        self.cartpole = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(URDF_PATH),
                fixed=True,
                default_armature=0.0,
            )
        )

        self.scene.build(n_envs=num_envs)

        # Joint indices from URDF
        self.cart_dof_idx = self.cartpole.get_joint("slider_to_cart").dofs_idx_local[0]
        self.pole_dof_idx = self.cartpole.get_joint("cart_to_pole").dofs_idx_local[0]
        self.dofs_idx = [self.cart_dof_idx, self.pole_dof_idx]

        self.reward_functions = {}
        self.episode_sums = {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(self.num_envs, device=self.device)

        # Initialize buffers.
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=gs.tc_int, device=self.device)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        # self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, 2), device=self.device)
        self.cart_position = torch.zeros(self.num_envs, device=self.device)
        self.cart_velocity = torch.zeros(self.num_envs, device=self.device)
        self.pole_angle = torch.zeros(self.num_envs, device=self.device)
        self.pole_velocity = torch.zeros(self.num_envs, device=self.device)
        self.extras = {}

        self.reset()

    def step(self, actions, disturbance_force=None):
        self.actions = torch.clip(actions, -1.0, 1.0)
        cart_force = self.actions * self.env_cfg["action_scale"]

        if disturbance_force is not None:
            cart_force = cart_force + disturbance_force

        self.cartpole.control_dofs_force(cart_force, dofs_idx_local=[self.cart_dof_idx])
        self.scene.step()

        # Update state buffers.
        self.episode_length_buf += 1
        self._update_state()

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            reward = reward_func() * self.reward_scales[name]
            self.rew_buf += reward
            self.episode_sums[name] += reward

    
        time_outs = self.episode_length_buf >= self.max_episode_length
        pole_fell = torch.abs(self.pole_angle) > math.radians(self.env_cfg["termination_angle_deg"])
        cart_out = torch.abs(self.cart_position) > self.env_cfg["termination_cart_position"]
        solver_error = self.scene.rigid_solver.get_error_envs_mask()
        self.reset_buf = time_outs | pole_fell | cart_out | solver_error

        self.extras = {"time_outs": time_outs.to(dtype=gs.tc_float)}
        done_envs = self.reset_buf.nonzero(as_tuple=False).flatten()

        if len(done_envs) > 0:
            self.extras["episode"] = {}
            for name in self.episode_sums.keys():
                self.extras["episode"]["rew_" + name] = self.episode_sums[name][done_envs].mean()

        dones = self.reset_buf.clone()
        self.reset_idx(done_envs)

        self._update_state()
        self._compute_observations()
        # self.last_actions[:] = self.actions

        return self.get_observations(), self.rew_buf, dones, self.extras

    def get_observations(self):
        return TensorDict(
            {"policy": self.obs_buf},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # theta_0 ~ Uniform(-20 degrees, +20 degrees).
        angle_limit = math.radians(self.env_cfg["initial_angle_deg"])
        initial_pole_angle = gs_rand_float(-angle_limit, angle_limit,
            (len(envs_idx),),
            self.device,
        )

        initial_dof_pos = torch.zeros((len(envs_idx), 2), device=self.device)
        initial_dof_pos[:, 1] = initial_pole_angle

        self.cartpole.set_dofs_position(
            position=initial_dof_pos,
            dofs_idx_local=self.dofs_idx,
            envs_idx=envs_idx,
            zero_velocity=True,
        )

        self.actions[envs_idx] = 0.0
        # self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        for name in self.episode_sums.keys():
            self.episode_sums[name][envs_idx] = 0.0

    def reset(self):
        envs_idx = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(envs_idx)
        self._update_state()
        self._compute_observations()
        return self.get_observations()

    def _update_state(self):
        self.dof_pos[:] = self.cartpole.get_dofs_position(self.dofs_idx)
        self.dof_vel[:] = self.cartpole.get_dofs_velocity(self.dofs_idx)

        self.cart_position[:] = self.dof_pos[:, 0]
        self.cart_velocity[:] = self.dof_vel[:, 0]

        # The pole joint is continuous, so keep its angle between -pi and +pi.
        self.pole_angle[:] = torch.atan2(
            torch.sin(self.dof_pos[:, 1]),
            torch.cos(self.dof_pos[:, 1]),
        )
        self.pole_velocity[:] = self.dof_vel[:, 1]

    def _compute_observations(self):
        self.obs_buf = torch.stack(
            (
                self.cart_position * self.obs_scales["cart_position"],
                self.cart_velocity * self.obs_scales["cart_velocity"],
                torch.sin(self.pole_angle),
                torch.cos(self.pole_angle),
                self.pole_velocity * self.obs_scales["pole_velocity"],
            ),
            dim=-1,
        )

    # Reward functions
    def _reward_upright(self):
        return torch.exp(-4.0 * torch.square(self.pole_angle))

    def _reward_cart_position(self):
        return torch.square(self.cart_position)

    def _reward_cart_velocity(self):
        return torch.square(self.cart_velocity)

    def _reward_pole_velocity(self):
        return torch.square(self.pole_velocity)

    def _reward_action(self):
        return torch.sum(torch.square(self.actions), dim=1)
