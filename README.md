# Cart-Pole Balancing with Genesis and PPO

This project trains a PPO reinforcement-learning policy to stabilize a cart-pole in the upright position using the [Genesis](https://genesis-world.readthedocs.io/) simulator and `rsl_rl`.

The policy pushes only the cart. The pole joint is passive, and an angle of zero represents the upright position. At every episode reset, the initial pole angle is sampled from:

\[
\theta_0 \sim \mathcal{U}(-20^\circ, 20^\circ)
\]

## Project structure

```text
CartPole-RL/
├── cartpole1.urdf       # Cart-pole model
├── cartpole_env.py      # Genesis environment, rewards, observations, and resets
├── cartpole_train.py    # PPO configuration and training entry point
├── cartpole_eval.py     # Policy evaluation and keyboard disturbances
└── requirements.txt     # Python dependencies
```

## Installation

Python 3.10 and a CUDA-capable GPU are recommended for training.

Create and activate a virtual environment, then install the dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build appropriate for your machine if the environment does not already provide one. The project was tested with Genesis `0.4.6` and `rsl-rl-lib` `5.0.1`.

## Training

Run from the repository directory:

```bash
python cartpole_train.py
```

Default training settings:

- Experiment name: `CartPole-RL1`
- Parallel environments: `4096`
- PPO iterations: `101`
- Backend: GPU
- Simulation timestep: `0.02 s`
- Episode duration: `25 s` (`1250` simulation steps)

The three available command-line arguments are:

```bash
python cartpole_train.py \
  --exp_name CartPole-RL1 \
  --num_envs 4096 \
  --max_iterations 101
```

Training creates the following directory:

```text
logs/CartPole-RL1/
├── cfgs.pkl
├── model_0.pt
├── model_50.pt
├── model_100.pt
├── events.out.tfevents.*
└── training_plots.png
```

`training_plots.png` contains the mean episode reward and mean episode length. The `logs/` directory is ignored by Git because checkpoints and event files can be large.

## Evaluation

Evaluate checkpoint 100 with:

```bash
python cartpole_eval.py --exp_name CartPole-RL1 --ckpt 100
```

Evaluation uses one environment and the CPU backend. Click the Genesis viewer, then use:

- **Left arrow:** apply a force disturbance to the left.
- **Right arrow:** apply a force disturbance to the right.

Close the viewer or press `Ctrl+C` to finish. The script saves:

```text
logs/CartPole-RL1/evaluation_plots_100.png
```

The evaluation plot shows pole angle, cart position, and keyboard disturbance force over time.

## Observations and action

The four physical states are cart position, cart velocity, pole angle, and pole angular velocity. PPO receives five observations:

```text
[scaled cart position,
 scaled cart velocity,
 sin(pole angle),
 cos(pole angle),
 scaled pole angular velocity]
```

The angle uses sine and cosine to avoid a discontinuity between `-pi` and `+pi`. The policy outputs one action in `[-1, 1]`, which is scaled into a horizontal force applied to the cart.

## Task termination

An episode ends when any of the following occurs:

- The configured episode duration is reached.
- The pole angle exceeds `60 degrees` from upright.
- The cart position exceeds `8 m` from the center.
- Genesis reports a solver error.
