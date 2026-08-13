import argparse
import os
import pickle
import shutil
import genesis as gs
from anymal_env import QuadV3
from performance_plots import save_training_plots
from rsl_rl.runners import OnPolicyRunner


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.6,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }
    return train_cfg_dict

def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        "urdf_path": "anymal_c_simple_description/urdf/anymal.urdf",
        "imu_link_name": "base",
        "imu_pos_offset": [0.2488, 0.00835, 0.04628],
        "enable_self_collision": False,
        "max_torques": [80.0] * 12,
        "default_joint_angles": {
            # Preload HAA inward so the loaded legs settle nearly vertical.
            "LF_HAA": -0.15, "RF_HAA": 0.15, "LH_HAA": -0.15, "RH_HAA": 0.15,
            "LF_HFE": 0.4, "RF_HFE": 0.4, "LH_HFE": -0.4, "RH_HFE": -0.4,
            # Straighter knees place the neutral base near the 0.50 m target.
            "LF_KFE": -0.5, "RF_KFE": -0.5, "LH_KFE": 0.5, "RH_KFE": 0.5,
        },
        # This is the same order as the 12 revolute joints in anymal.urdf.
        "joint_names": [
            "LF_HAA", "LF_HFE", "LF_KFE",
            "RF_HAA", "RF_HFE", "RF_KFE",
            "LH_HAA", "LH_HFE", "LH_KFE",
            "RH_HAA", "RH_HFE", "RH_KFE",
        ],
        "kp": 80.0,
        "kd": 2.0,
        "termination_if_roll_greater_than": 60, # Ends training if robot tilts sideways x degrees.
        "termination_if_pitch_greater_than": 60, # Ends if robot tilts forward/backward x degrees
        # Spawn above the nominal stance so the feet do not begin inside the ground.
        "base_init_pos": [0.0, 0.0, 0.58],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0], # starting rotation in Quaternions
        "termination_if_height_lower_than": 0.25,
        "episode_length_s": 20.0,
        "resampling_time_s": 8.0, # Every n sec, the robot is given a new random velocity command
        "action_scale": 0.5,
        "clip_actions": 1.0,
        "zero_command_probability": 0.20,
        "straight_command_probability": 0.40,
    }
    obs_cfg = {
        "num_obs": 45, # gyro, gravity, command, joint state and previous action
        "obs_scales": {
            "lin_vel": 2.0, # Scales the linear velocity
            "ang_vel": 0.25, # Shrinks the raw angular velocity values (how fast it spins)
            "dof_pos": 1.0, # Keeps the "Degrees of Freedom" positions (joint angles) at their original scale (usually radians).
            "dof_vel": 0.05, # reduces the raw joint speed values
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.20,
        "base_height_target": 0.55, 
        "reward_scales": {
            "tracking_lin_vel": 1.0, 
            "tracking_ang_vel": 0.2, 
            "lin_vel_z": -1.0, 
            "base_height": -50.0, 
            "action_rate": -0.01, 
            "similar_to_default": -0.1, # dec
            "ang_vel_xy": -0.05, 
            "orientation": -1.0, 
            "torques": -2.0e-5,
            "dof_acc": -2.5e-7, 
        },
    }
    command_cfg = {
        "num_commands": 3, # control inputs (Forward/Backward, Strafe Left/Right, and Turn).
        "lin_vel_x_range": [-1.0, 1.0], # Walk forward/backward
        "lin_vel_y_range": [-0.5, 0.5], # side-to-side (strafing) movement
        "ang_vel_range": [-1.0, 1.0], # Turning
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="Anymal1")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=301)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    if args.max_iterations < 1:
        parser.error("--max_iterations must be positive")
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/cfgs.pkl", "wb") as cfg_file:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], cfg_file)

    env = QuadV3(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg, 
        show_viewer=False
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    print(f"Starting {args.max_iterations} iterations of ANYmal C locomotion training")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    writer = getattr(runner.logger, "writer", None)
    if writer is not None:
        writer.flush()
    save_training_plots(log_dir)

if __name__ == "__main__":
    main()
