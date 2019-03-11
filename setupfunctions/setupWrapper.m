function [imu, smk, cmd, CF] = setupWrapper()
% Wrapper sets up the snake monster for use in the CPG code.

disp('Setting up snake monster...');

load('names.mat');
shoulders = names(1:3:end);
imu = HebiLookup.newGroupFromNames('*',shoulders);
snakeData = setupSnakeMonsterShoulderData();
smk = SnakeMonsterKinematics();

load('offsets.mat');
setupSnakeMonster();
cmd = CommandStruct();
CF = SMComplementaryFilter(snakeData, 'accelOffsets', accelOffsets, 'gyroOffsets', gyroOffsets);
CF.update(imu.getNextFeedback());

disp('Setup complete!');
end

