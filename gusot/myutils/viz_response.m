function [out] = viz_response(mat)
%VIZ_RESPONSE Summary of this function goes here
%   Detailed explanation goes here
[m,n] = size(mat);
out = circshift(mat,[round(m/2), round(n/2)]);
end

