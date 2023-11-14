function [patch,scale] = getPatch_m(im,bbox,size_z,size_x,context,fill_color)
%GETPATCH_M Summary of this function goes here
%   Detailed explanation goes here
if nargin<5
    context = 0;
end
if nargin<6
    [patch,scale] = get_crops_m(im, convert_bbox_format_m(bbox,'center'), size_z, size_x, context);
else
    [patch,scale] = get_crops_m(im, convert_bbox_format_m(bbox,'center'), size_z, size_x, context,fill_color);
end

end

