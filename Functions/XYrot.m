function [T,newPose] = XYrot(pose)

[~,~,gamma] = decomposeSO3(pose);
newPose = rotz(gamma);
newPose = newPose(1:3,1:3);

y1 = pose * [0;1;0];
y2 = newPose * [0;1;0];

rotM = rotz( atan2(y1(2), y1(1)) - atan2(y2(2), y2(1)) );
newPose = rotM(1:3,1:3) * newPose;
T = newPose / pose;
