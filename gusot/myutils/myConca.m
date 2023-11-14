function [out] = myConca(batch,conca_wind)
%MYCONCA Summary of this function goes here
%   if spatial sizes are odd, the last spatial element will be discarded
m = size(batch,1);
n = size(batch,2);
if mod(m,2)>0
    m = m-1;
end
if mod(n,2)>0
    n = n-1;
end

mat = cell(4,1);
mat{1} = batch(1:conca_wind:m, 1:conca_wind:n, :, :);
mat{2} = batch(2:conca_wind:m, 1:conca_wind:n, :, :);
mat{3} = batch(2:conca_wind:m, 2:conca_wind:n, :, :);
mat{4} = batch(1:conca_wind:m, 2:conca_wind:n, :, :);
out = cat(3,mat{:});

end

