import argparse
import pickle
import shutil
from pathlib import Path
from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from cartpole_env import CartPoleEnv
SCRIPT_DIR = Path(__file__).resolve().parent


def save_training_plots(log_dir):
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_data = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    event_data.Reload()
    available_tags = event_data.Tags()["scalars"]

    def plot_metric(axis, tag, title, ylabel):
        if tag in available_tags:
            values = event_data.Scalars(tag)
            axis.plot([value.step for value in values], [value.value for value in values])
        else:
            axis.text(0.5, 0.5, "No data recorded", ha="center", va="center")

        axis.set_title(title)
        axis.set_xlabel("PPO iteration")
        axis.set_ylabel(ylabel)
        axis.grid(True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_metric(axes[0], "Train/mean_reward", "Mean episode reward", "Reward")
    plot_metric(axes[1], "Train/mean_episode_length", "Mean episode length", "Steps")

    figure.suptitle("Cart-pole PPO training results")
    figure.tight_layout()
    plot_path = log_dir / "training_plots.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    print(f"Training plots saved to: {plot_path}")


def get_train_cfg(exp_name):
    train_cfg_dict = {
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
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [64, 64],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [64, 64],
            "activation": "elu",
            "obs_normalization": False,
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 32,
        "save_interval": 50,
        "run_name": exp_name,
        "logger": "tensorboard",
    }
    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 1,
        "episode_length_s": 25.0,
        "action_scale": 100.0,
        "termination_angle_deg": 60.0,
        "termination_cart_position": 8.0,
        "initial_angle_deg": 20.0,
    }

    obs_cfg = {
        "num_obs": 5,
        "obs_scales": {
            "cart_position": 0.125,
            "cart_velocity": 0.2,
            "pole_velocity": 0.1,
        },
    }

    reward_cfg = {
        "reward_scales": {
            "upright": 1.0,
            "cart_position": -0.01,
            "cart_velocity": -0.001,
            "pole_velocity": -0.001,
            "action": -0.01
        },
    }

    return env_cfg, obs_cfg, reward_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="CartPole-RL1")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=101)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level="warning")

    log_dir = SCRIPT_DIR / "logs" / args.exp_name
    env_cfg, obs_cfg, reward_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name)

    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True)

    with (log_dir / "cfgs.pkl").open("wb") as file:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, train_cfg], file)

    env = CartPoleEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=False,
    )

    print(f"Training started on {args.num_envs} environments...")
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    save_training_plots(log_dir)

if __name__ == "__main__":
    main()
