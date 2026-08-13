function cartpole_sim(x, theta)

persistent fig ax cart pole bob

% Visualization dimensions
cartWidth  = 0.4;
cartHeight = 0.2;

% l = 0.5 m is pivot-to-COM.
% Use 1.0 m as the visual full pole length.
L = 1.0;

pivotY = cartHeight/2;

%% Create animation window the first time
if isempty(fig) || ~isvalid(fig)

    fig = figure( ...
        'Name','Cart-Pole MPC Animation', ...
        'NumberTitle','off');

    ax = axes(fig);

    hold(ax,'on');
    grid(ax,'on');
    axis(ax,'equal');

    xlim(ax,[-2 2]);
    ylim(ax,[-0.5 1.5]);

    xlabel(ax,'Cart Position [m]');
    ylabel(ax,'Height [m]');
    title(ax,'MPC Cart-Pole');

    % Ground / rail
    plot(ax,[-10 10],[-0.12 -0.12],'k','LineWidth',2);

    % Cart
    cart = rectangle(ax, ...
        'Position',[x-cartWidth/2, -cartHeight/2, ...
                    cartWidth, cartHeight], ...
        'FaceColor',[0.2 0.5 0.8]);

    % Initial pole
    xp = x + L*sin(theta);
    yp = pivotY + L*cos(theta);

    pole = plot(ax,[x xp],[pivotY yp], ...
        'LineWidth',4);

    % Pole tip
    bob = plot(ax,xp,yp,'o', ...
        'MarkerSize',10, ...
        'MarkerFaceColor','r');

end

%% Calculate current pole position
xp = x + L*sin(theta);
yp = pivotY + L*cos(theta);

%% Update cart
set(cart,'Position', ...
    [x-cartWidth/2, -cartHeight/2, ...
     cartWidth, cartHeight]);

%% Update pole
set(pole, ...
    'XData',[x xp], ...
    'YData',[pivotY yp]);

%% Update tip
set(bob, ...
    'XData',xp, ...
    'YData',yp);

%% Keep camera centered if cart moves far
xlim(ax,[x-2 x+2]);

drawnow limitrate

end