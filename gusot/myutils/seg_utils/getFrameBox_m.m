function [out] = getFrameBox_m(bb,scale,prev_bb,patch_shape)
%GETFRAMEBOX Summary of this function goes here
%   Detailed explanation goes here
out = [];
if ~isempty(bb)
    tmp = convert_bbox_format_m(prev_bb, 'center');
    cx = tmp(1);
    cy = tmp(2);
    cx_patch = get_center_m(patch_shape(2));
    cy_patch = get_center_m(patch_shape(1));
    xmin = (bb(1)-cx_patch)/scale + cx;
    xmax = (bb(1)+bb(3)-1-cx_patch)/scale + cx;
    ymin = (bb(2)-cy_patch)/scale + cy;
    ymax = (bb(2)+bb(4)-1-cy_patch)/scale + cy;
    out = [xmin,ymin,xmax-xmin+1,ymax-ymin+1];
end

end

