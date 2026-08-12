# Cart-Pole Balancing with Genesis and PPO

This project trains a PPO reinforcement-learning policy to stabilize a cart-pole in the upright position using the [Genesis](https://genesis-world.readthedocs.io/) simulator and `rsl_rl`.

A trained PPO checkpoint is included, so evaluation can be run immediately after cloning and installing the dependencies.

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

Python 3.10 is recommended. A CUDA-capable GPU is recommended for training, while evaluation runs on the CPU.

Cloning the repository does not install Python or the required packages. Running the scripts with an unprepared system Python can produce errors such as `python: command not found` or `No module named 'typing_extensions'`.

First, clone the repository:

```bash
git clone https://github.com/abdullah-tm14/CartPole-MPC-RL.git
cd CartPole-MPC-RL
```

Then use either Conda or `venv`. Do not run both setup methods.

### Option 1: Conda

If you already have a suitable Conda environment, activate it before running the project:

```bash
conda activate rl
python -m pip install -r requirements.txt
```

To create a new environment instead:

```bash
conda create -n cartpole python=3.10 -y
conda activate cartpole
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option 2: Python virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After activation, verify that `python` belongs to the selected environment:

```bash
which python
```

### Installing Genesis manually

Genesis is installed automatically by `requirements.txt`. If Genesis is missing, install the tested version directly:

```bash
python -m pip install genesis-world==0.4.6
```

You can confirm the installation with:

```bash
python -m pip show genesis-world
```

The project was tested with Genesis `0.4.6` and `rsl-rl-lib` `5.0.1`. For GPU training, install a CUDA-enabled PyTorch build appropriate for your CUDA version. The included evaluation uses the CPU backend.

## Run the trained policy

With the environment still activated, run:

```bash
python cartpole_eval.py
```

This loads the included checkpoint:

```text
logs/CartPole-RL1/model_100.pt
```

The Genesis viewer opens with one cart-pole. Click the viewer and use the left and right arrow keys to apply force disturbances. Close the viewer or press `Ctrl+C` to stop and save the evaluation plot.

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

`training_plots.png` contains the mean episode reward and mean episode length. The included `logs/` directory contains the trained checkpoints, saved configurations, TensorBoard event data, and result plots.

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
