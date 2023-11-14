function [out] = convert_bbox_format_m(bbox,to)
%CONVERT_BBOX_FORMAT Summary of this function goes here
%   Detailed explanation goes here
x = bbox(1);
y = bbox(2);
w = bbox(3);
h = bbox(4);
if strcmp(to,'topleft')
    x = x-get_center_m(w);
    y = y-get_center_m(h);
elseif strcmp(to,'center')
    x = x+get_center_m(w);
    y = y+get_center_m(h);
else
    ME = MException('seg_utils:convert_bbox_format', ...
        'value for [to] = %s not defined',to);
    throw(ME);
end
out = [x,y,w,h];
end

