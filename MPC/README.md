# Cart-Pole Model Predictive Control

This project stabilizes a linearized cart-pole around the upright equilibrium with model predictive control (MPC) in MATLAB and Simulink. The controller regulates cart position, cart velocity, pole angle, and pole angular velocity while constraining the horizontal control force.

## Requirements

- MATLAB (the included model was created with R2022b)
- Simulink
- Control System Toolbox
- Model Predictive Control Toolbox

## Files

```text
MPC/
├── CartPole.m             # Plant model, discretization, MPC tuning, and initial state
├── untitled.slx           # Simulink closed-loop model
├── cartpole_sim.m         # Cart-pole animation function
├── plots.m                # Plots logged simulation outputs
└── Results and plots/     # Diagram, response plots, animation frames, and video
```

Generated `slprj/` data and `.slxc` cache files are not required and are intentionally excluded from version control.

## Run after cloning

Clone the repository, start MATLAB, and make `MPC/` the current folder:

```matlab
cd('/path/to/CartPole-MPC-RL/MPC')
```

Initialize the plant and controller, then open and simulate the model:

```matlab
run('CartPole.m')
open_system('untitled.slx')
out = sim('untitled.slx', 'StopTime', num2str(Tsim));
```

The simulation starts with a reproducible random pole angle between `-20` and `20` degrees. The Simulink model uses `cartpole_sim.m` for the animation and returns the state and control histories in `out.yout` and `out.uout`.

Generate the result figures after the simulation:

```matlab
run('plots.m')
```

You can also click **Run** in Simulink after executing `CartPole.m`; assign the simulation output to `out` if you want to use `plots.m` unchanged.

## Model and controller

The continuous state is:

```text
x = [cart position, cart velocity, pole angle, pole angular velocity]
```

`CartPole.m` defines the linearized continuous-time state-space plant and discretizes it with zero-order hold at `Ts = 0.02 s` (50 Hz). The MPC configuration uses:

- Prediction horizon: `50` samples (`1 s`)
- Control horizon: `10` samples
- Output weights: `[20, 1, 60, 5]`
- Manipulated-variable weight: `0.01`
- Manipulated-variable-rate weight: `0.5`
- Force constraint: `-10 N <= u <= 10 N`
- Simulation duration: `10 s`

The reference is the upright equilibrium with the cart centered and all velocities equal to zero.

## Included results

`Results and plots/` contains the Simulink diagram, closed-loop response figures, cart-pole animation images, and `CartPole-MPC.mp4`.

![MPC response](Results%20and%20plots/response.jpg)
