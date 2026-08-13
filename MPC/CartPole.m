clear;
clc;
close all;

M = 1.0;
m = 0.2; 
l = 0.5;
g = 9.81;

% Standard CartPole Dynamics
A = [0  1   0                  0;
     0  0  -m*g/M              0;
     0  0   0                  1;
     0  0   (M+m)*g/(M*l)      0];

B = [0;
     1/M;
     0;
    -1/(M*l)];

C = eye(4);
D = zeros(4,1);

% SS plant
plant_c = ss(A,B,C,D);

% Sampling time (50 Hz)
Ts = 0.02;

% Discretize plant
plant_d = c2d(plant_c,Ts,'zoh');

Ad = plant_d.A;
Bd = plant_d.B;
Cd = plant_d.C;
Dd = plant_d.D;

plant_dd = ss(Ad,Bd,Cd,Dd);

% MPC cont

% how far plant behavior is predicted
prediction_horizon = 50; % 50(0.02) -> controller predicts every 1s

% how many future control moves are optimized
control_horizon = 10; % determines how many future control moves

mpcobj = mpc(plant_d,Ts, prediction_horizon, control_horizon);

% MPC weights

% % [Position, Velocity, Angle, Ang_Vel]
mpcobj.Weights.OutputVariables = [20 1 60 5]; 

% Penalize large force
mpcobj.Weights.ManipulatedVariables = 0.01;

% Penalize sudden changes in force
mpcobj.Weights.ManipulatedVariablesRate = 0.5;

% force limits
mpcobj.MV.Min = -10;
mpcobj.MV.Max =  10;

% random initial angle
rng(4,'twister');

theta0_deg = -20 + 40*rand; % rand -> [0,1]
theta0 = deg2rad(theta0_deg);

% Other initial states chosen as zero
x0 = [0; 0; theta0; 0];

% Reference (Goal)
ref = [0; 0; 0; 0];

Tsim = 10;

fprintf('Initial pole angle = %.2f degrees\n',theta0_deg);