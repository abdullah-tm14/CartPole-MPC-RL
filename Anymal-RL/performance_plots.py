from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _save_line_plot(x, y, xlabel, ylabel, title, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_training_plots(log_dir):
    """Save the two required training plots from the latest TensorBoard log."""
    log_dir = Path(log_dir)
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in {log_dir}")

    event_file = max(event_files, key=lambda path: path.stat().st_mtime)
    events = EventAccumulator(str(event_file))
    events.Reload()

    plot_dir = log_dir / "performance"
    specifications = (
        (
            "Train/mean_reward",
            "Mean episode reward",
            "Training reward progression",
            plot_dir / "training_reward_progression.png",
        ),
        (
            "Train/mean_episode_length",
            "Mean episode length (steps)",
            "Episode length progression",
            plot_dir / "episode_length_progression.png",
        ),
    )

    available_tags = events.Tags().get("scalars", [])
    for tag, ylabel, title, output_path in specifications:
        if tag not in available_tags:
            raise KeyError(f"TensorBoard tag {tag!r} was not written during training")
        scalar_events = events.Scalars(tag)
        _save_line_plot(
            [event.step for event in scalar_events],
            [event.value for event in scalar_events],
            "Training iteration",
            ylabel,
            title,
            output_path,
        )

    print(f"Saved training plots to {plot_dir}")


def save_velocity_tracking_plot(output_dir, times, commanded_velocities, actual_velocities):
    """Save commanded and measured forward velocity from an evaluation run."""
    if not times:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation_velocity_tracking.png"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times, commanded_velocities, label="Commanded velocity", linewidth=2)
    ax.plot(times, actual_velocities, label="Actual velocity", linewidth=1.5)
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Forward velocity (m/s)")
    ax.set_title("Evaluation velocity tracking")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"Saved evaluation plot to {output_path}")
    return output_path
