function [value,flag_out] = getValue_my(mat,idx1,idx2)
%GETVALUE Summary of this function goes here
%   Detailed explanation goes here
[m,n] = size(mat);
row = round(idx1);
col = round(idx2);
flag_out = false;
if row<1 || row>m || col<1 || col>n
    value = 0;
    flag_out = true;
else
    value = mat(row,col);
end
end

