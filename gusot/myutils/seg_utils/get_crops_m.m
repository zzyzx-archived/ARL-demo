function [image_crop_x, scale_x] = get_crops_m(im, bbox, size_z, size_x, context_amount,fill_color)
%GET_CROPS_M Summary of this function goes here
%   Detailed explanation goes here
cy = bbox(2);
cx = bbox(1);
h = bbox(4);
w = bbox(3);
wc_z = w + context_amount*(w+h);
hc_z = h + context_amount*(w+h);
s_z = sqrt(wc_z*hc_z);
scale_z = size_z / s_z;

d_search = (size_x - size_z) / 2;
pad = d_search / scale_z;
base_s_x = s_z + 2 * pad;
s_x = base_s_x;
base_scale_x = size_x / base_s_x;
scale_x = base_scale_x;

if nargin<6
    image_crop_x = get_subwindow_avg_m(im, [cy, cx], [size_x, size_x], [round(s_x) round(s_x)]);
else
    image_crop_x = get_subwindow_avg_m(im, [cy, cx], [size_x, size_x], [round(s_x) round(s_x)], fill_color);
end

end

