import argparse
import pickle
from pathlib import Path
import torch
from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind
from cartpole_env import CartPoleEnv


SCRIPT_DIR = Path(__file__).resolve().parent # to read the urdf
DISTURBANCE_FORCE = 50.0
DISTURBANCE_STEPS = 5


def save_evaluation_plots(log_dir, checkpoint, time_values, angles, positions, disturbances):
    if len(time_values) == 0:
        print("No evaluation data was recorded, so no plot was saved.")
        return

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(time_values, angles)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("Pole angle")

    axes[1].plot(time_values, positions)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Position (m)")
    axes[1].set_title("Cart position")

    axes[2].plot(time_values, disturbances, color="red")
    axes[2].set_ylabel("Force (N)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Disturbance force")

    for axis in axes:
        axis.grid(True)

    figure.suptitle(f"Cart-pole evaluation: checkpoint {checkpoint}")
    figure.tight_layout()
    plot_path = log_dir / f"evaluation_plots_{checkpoint}.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    print(f"Evaluation plots saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="CartPole-RL1")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")

    log_dir = SCRIPT_DIR / "logs" / args.exp_name
    with (log_dir / "cfgs.pkl").open("rb") as file:
        env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(file)

    # reward_cfg["reward_scales"] = {}

    env = CartPoleEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    resume_path = log_dir / f"model_{args.ckpt}.pt"
    runner.load(str(resume_path))
    policy = runner.get_inference_policy(device=gs.device)
    disturbance = {"force": 0.0, "steps_left": 0}

    def push_cart(force):
        disturbance["force"] = force
        disturbance["steps_left"] = DISTURBANCE_STEPS

    env.scene.viewer.register_keybinds(
        Keybind("push_left", Key.LEFT, KeyAction.PRESS, callback=push_cart, args=(-DISTURBANCE_FORCE,)),
        Keybind("push_right", Key.RIGHT, KeyAction.PRESS, callback=push_cart, args=(DISTURBANCE_FORCE,)),
    )

    print("Keyboard disturbance:")
    print("  LEFT ARROW  = push the cart to the left")
    print("  RIGHT ARROW = push the cart to the right")

    obs = env.reset()
    time_values = []
    angles = []
    positions = []
    disturbances = []
    step_count = 0

    try:
        with torch.no_grad():
            while env.scene.viewer.is_alive():
                actions = policy(obs)

                if disturbance["steps_left"] > 0:
                    disturbance_value = disturbance["force"]
                    external_force = torch.tensor([[disturbance_value]], device=gs.device)
                    disturbance["steps_left"] -= 1
                else:
                    disturbance_value = 0.0
                    external_force = None

                obs, rews, dones, infos = env.step(actions, disturbance_force=external_force)

                step_count += 1
                time_values.append(step_count * env.dt)
                angles.append(torch.rad2deg(env.pole_angle[0]).item())
                positions.append(env.cart_position[0].item())
                disturbances.append(disturbance_value)
    except KeyboardInterrupt:
        print("Evaluation stopped by the user.")
    finally:
        save_evaluation_plots(
            log_dir,
            args.ckpt,
            time_values,
            angles,
            positions,
            disturbances,
        )


if __name__ == "__main__":
    main()
