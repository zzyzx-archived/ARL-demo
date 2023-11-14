function [dif] = fixBorder_(dif,dx,dy)
%FIXBORDER Summary of this function goes here
%   Detailed explanation goes here
h = size(dif,1);
w = size(dif,2);
if dx>=0
    xmin = 1;
    xmax = min(w,round(dx)+1);
else
    xmin = max(1,round(w-1+dx-5));
    xmax = w;
end

if dy>=0
    ymin = 1;
    ymax = min(h,round(dy)+1);
else
    ymin = max(1,round(h-1+dy-5));
    ymax = h;
end

dif(int32(ymin):int32(ymax),:) = 0;
dif(:,int32(xmin):int32(xmax)) = 0;
end

