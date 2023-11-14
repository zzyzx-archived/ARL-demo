function [out] = getScaledBox_m(prev_bb,gt_bb,scale,patch_shape)
%GETSCALEDBOX Summary of this function goes here
%   Detailed explanation goes here
tmp = convert_bbox_format_m(prev_bb, 'center');
cx = tmp(1);
cy = tmp(2);
cx_patch = get_center_m(patch_shape(2));
cy_patch = get_center_m(patch_shape(1));
xmin = scale*(gt_bb(1)-cx)+cx_patch;
xmax = scale*(gt_bb(1)+gt_bb(3)-1-cx)+cx_patch;
ymin = scale*(gt_bb(2)-cy)+cy_patch;
ymax = scale*(gt_bb(2)+gt_bb(4)-1-cy)+cy_patch;
out = [xmin,ymin,xmax-xmin+1,ymax-ymin+1];

end

