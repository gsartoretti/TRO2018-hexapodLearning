function [theta] = jacobianAngleCorrection(cpg)
%% Calculate Angle Correction

J = cpg.smk.getLegJacobians(cpg.legs);

theta = zeros(3,6);
numLegs = 6;

for leg = 1:numLegs
    theta(:,leg) = J(1:3,:,leg) \ cpg.dx(:,leg);     
end

end

