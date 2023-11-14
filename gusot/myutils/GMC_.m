function [success,m, dif] = GMC_(prev,curr)
%GMC Summary of this function goes here
%   Detailed explanation goes here
success = false;
m = [];
dif = [];
verbose = false;

if size(prev,3)>1
    prev_gray = rgb2gray(prev);
else
    prev_gray = prev;
end

if size(curr,3)>1
    curr_gray = rgb2gray(curr);
else
    curr_gray = curr;
end

ptThresh = 0.2;
pointsA = detectFASTFeatures(prev_gray, 'MinContrast', ptThresh).selectStrongest(200);
pointsB = detectFASTFeatures(curr_gray, 'MinContrast', ptThresh).selectStrongest(200);
% Extract FREAK descriptors for the corners
[featuresA, pointsA] = extractFeatures(prev_gray, pointsA);
[featuresB, pointsB] = extractFeatures(curr_gray, pointsB);

indexPairs = matchFeatures(featuresA, featuresB);
pointsA = pointsA(indexPairs(:, 1), :);
pointsB = pointsB(indexPairs(:, 2), :);
if verbose
    figure; showMatchedFeatures(prev_gray, curr_gray, pointsA, pointsB);
    legend('A', 'B');
end

[tform, inlierIdx] = estimateGeometricTransform2D(...
    pointsA, pointsB, 'affine');
% pointsBm = pointsB(inlierIdx, :);
% pointsAm = pointsA(inlierIdx, :);

m = tform.T;

if ~isempty(m)
    dx = m(3,1);
    dy = m(3,2);
    da = atan2(m(1,2), m(1,1));

    if abs(da)<0.01
        gray_stabilized = imwarp(prev_gray, tform, 'OutputView', imref2d(size(prev_gray)));
        %pointsAmp = transformPointsForward(tform, pointsAm.Location);
        dif = abs(double(curr_gray)-double(gray_stabilized));
        dif = fixBorder_(dif,dx,dy);

        % th to clean
        dif(dif<(mean(dif,'all')+30)) = 0;
    end
end

end

