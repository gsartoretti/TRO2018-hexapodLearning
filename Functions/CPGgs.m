function [cpg] = CPGgs(cpg, t, dt)


theta = zeros(3,6);

if t > cpg.initLength
     
    [cpg] = stabilizationPID(cpg);
      
    theta = jacobianAngleCorrection(cpg);

else
    cpg.legs = zeros(1,18);
end


%Filter large spikes in theta 
theta(abs(theta) > 1.0) = 0.0;

%shoulder2Offsets    = pi/8 * ones(1,6); %flat rocks
shoulder2Offsets    = pi/5 * ones(1,6); %tilt

%Move back to uncorrected stance
alpha = 0.1;
beta = 0.1;

theta(2,:) = theta(2,:) - alpha * cpg.theta2;
theta(3,:) = theta(3,:) - beta * cpg.theta3;



%% Applies CPG to the first two joints and IK to the last joint.

% height
% a = 0.5 * [.3 .3 .3 .3 .3 .3 ];%[.3 .3 .3 .3 .3 .3]; % semi major-axis of the limit-ellipse %flat rocks
a = 0.6 * ones(1, 6);%[.3 .3 .3 .3 .3 .3]; % semi major-axis of the limit-ellipse  %tilt

% length
 b = pi/18 * ones(1, 6); % semi minor-axis of the limit-ellipse

%step height
shoulders1          = 1:3:18; % joint IDs of the shoulders
shoulders1Corr      = [1 -1 1 -1 1 -1] .* cpg.direction; % correction factor for left/right legs
shoulder1Offsets    = [-1 -1 0 0 1 1] * pi/3 .* cpg.direction; % offset so that legs are more spread out
shoulders2          = 2:3:18; % joint IDs of the second shoulder joints
elbows              = 3:3:18; % joint IDs of the elbow joints

% Robot Dimensions
endWidth = 0.075; % dimensions of the end section of the leg
endHeight = 0.185;
endTheta = atan(endHeight/endWidth);
L1 = 0.125; % all distances in m
L2 = sqrt(endWidth^2 + endHeight^2);
moduleLen = .097;

xKy = 0.08; % distance of the central leg from the shoulder
GSoffset = 0.07;
gammaX = 20;
gammaY = 20;
sweep = pi/24;


radCentral = L1*cos(shoulder2Offsets(1)) + .063-.0122;
d = 2 * tan(sweep./(2*b)) * (moduleLen + radCentral);
r0 = moduleLen + [xKy,xKy,radCentral,radCentral,xKy,xKy] - GSoffset;

K = [ 0 -1 -1  1  1 -1;
     -1  0  1 -1 -1  1;
     -1  1  0 -1 -1  1;
      1 -1 -1  0  1 -1;
      1 -1 -1  1  0 -1;
     -1  1  1 -1 -1  0];

% CPG Equations
dx = gammaX .* (1- (cpg.x(t,:).^2)./(b.^2) - ((cpg.y(t,:) - cpg.theta2).^2)./(a.^2)).*cpg.x(t, :) - cpg.w_y .* b ./ a .* (cpg.y(t, :) - cpg.theta2);
dy = gammaY .* (1- (cpg.x(t,:).^2)./(b.^2) - ((cpg.y(t,:) - cpg.theta2).^2)./(a.^2)).*(cpg.y(t,:) - cpg.theta2) + cpg.w_y .* a ./ b .* cpg.x(t, :) + (K*(cpg.y(t,:) - cpg.theta2)')'./4 + theta(2,:);

cpg.theta2 = cpg.theta2 + theta(2,:) * dt;
cpg.theta3 = cpg.theta3 + theta(3,:) * dt;

if ~cpg.move
    dx = 0;
    dy = theta(2,:);
end

cpg.x(t+1,:) = cpg.x(t,:) + dx * dt;
cpg.y(t+1,:) = cpg.y(t,:) + dy * dt;

%% CPG
r0s = r0 * xKy / r0(3);

cpg.legs(shoulders1) = (shoulder1Offsets+cpg.x(t+1,:)) .* shoulders1Corr; %CPG Controlled
cpg.legs(shoulders2) = (shoulder2Offsets+max(cpg.theta2,cpg.y(t+1,:)));
cpg.legs(elbows) = ( asin( (r0s./cos(cpg.legs(shoulders1)) - L1.*cos(cpg.legs(shoulders2)))/L2) ...
             - cpg.legs(shoulders2) - pi/2 + endTheta) + cpg.theta3;


end

