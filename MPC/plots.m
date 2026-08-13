% simulation results
t = out.yout.Time;
Y = squeeze(out.yout.Data);

tu = out.uout.Time;
U = squeeze(out.uout.Data);

position = Y(:,1);
velocity = Y(:,2);
angle = Y(:,3);
angular_velocity = Y(:,4);

angle_deg = rad2deg(angle);

% Pole Angle - Alpha
figure;
plot(t,angle_deg,'LineWidth',1.5);
grid on;
xlabel('Time [s]');
ylabel('Pole Angle [deg]');
title('Pole Angle - Alpha');
yline(0,'--');

% Cart Position - x
figure;
plot(t,position,'LineWidth',1.5);
grid on;
xlabel('Time [s]');
ylabel('Cart Position [m]');
title('Cart Position - x');
yline(0,'--');

% Pole Angular Velocity - Alpha dot
figure;
plot(t,angular_velocity,'LineWidth',1.5);
grid on;
xlabel('Time [s]');
ylabel('Angular Velocity [rad/s]');
title('Pole Angular Velocity - Alpha dot');
yline(0,'--');

% Control Force - u
figure;
plot(tu,U,'LineWidth',1.5);
grid on;
xlabel('Time [s]');
ylabel('Control Force [N]');
title('Control Input - u');
yline(20,'--');
yline(-20,'--');