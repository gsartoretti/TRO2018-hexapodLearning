  function [ snakeData ] = setupSnakeMonsterShoulderData()
%SETUPSNAKEMONSTERDATA Initializes the data needed to run the matlab control code
%on one of our snakes.
%
%   The hope is that using this function will allow us to design code that
%   can be used on future snakes with different configurations, with a
%   minimal amount of re-work.
%
%   [snakeData] = setupSnakeMonsterData( snakeType, numModules )
%       Returns a struct based on the type of snake and the number of 
%       modules specified.  THe snake data strcut contains:
%           .snakeType
%
%           .modules
%           .numModules
%
%           .moduleLen
%           .moduleDia
%
%           .axisPerm
%
%           .snakeShape
%
% Dave Rollinson
% Oct 2013

    % String Describing Snake Type
    snakeData.snakeType = 'Snake Monster';

    % Number of modules
    numModules = 6;
    snakeData.modules = 1:numModules;
    snakeData.num_modules = numModules;

    %%%%%%%%%%%%%%%%%%
    % KINEMATIC INFO %
    %%%%%%%%%%%%%%%%%%
    % Length of modules
    snakeData.moduleLen = .0639;    % meters
    % Diameter of modules 
    % (for animation only, does not effect kinematics)
    % (may come into play for motion models)
    snakeData.moduleDia = .0508;    % meters (no skin)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Snake Monster module degrees of freedom %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1 3 5 -> same direction, 2 4 6 -> turned pi around Y axis
    %  Leg Numbering / Chassis Coordinate convention:
    %  
    %   2 ----- 1     +y
    %       |          ^
    %   4 ----- 3      |
    %       |          o--> +x
    %   6 ----- 5    +z
    
    % Rotation Matrices
    R_x = @(t)[1 0 0; 0 cos(t) -sin(t); 0 sin(t) cos(t)];
    R_y = @(t)[cos(t) 0 sin(t); 0 1 0; -sin(t) 0 cos(t)];
    R_z = @(t)[cos(t) -sin(t) 0; sin(t) cos(t) 0; 0 0 1];
    
    snakeData.snakeShape = zeros(4,4,7);
    snakeData.snakeShape(1:3,1:3,2) = R_y(pi/2) * R_z(-pi/2); % should joint 1
    snakeData.snakeShape(1:3,1:3,3) = R_y(pi/2) * R_z(-pi/2) * R_y(-pi); % should joint 2
    snakeData.snakeShape(1:3,1:3,4) = snakeData.snakeShape(1:3,1:3,2); % should joint 3
    snakeData.snakeShape(1:3,1:3,5) = snakeData.snakeShape(1:3,1:3,3); % should joint 4
    snakeData.snakeShape(1:3,1:3,6) = snakeData.snakeShape(1:3,1:3,2); % should joint 5
    snakeData.snakeShape(1:3,1:3,7) = snakeData.snakeShape(1:3,1:3,3); % should joint 6
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Permutation Axis for Virtual Chassis %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Defaults to X-Y-Z being 1st-2nd-3rd Principal Moments
    snakeData.axisPerm = [ 1  0  0
                           0  1  0
                           0  0  1 ];
                       
%     snakeData.firmwareType = firmware;

end

