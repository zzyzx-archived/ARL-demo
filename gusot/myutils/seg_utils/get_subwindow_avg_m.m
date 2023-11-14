function [im_patch] = get_subwindow_avg_m(im, pos, model_sz, original_sz,fill_color)
%GET_SUBWINDOW_AVG_M Summary of this function goes here
%   Detailed explanation goes here
if nargin<5
   fill_color = [mean(im(:,:,1),'all'), mean(im(:,:,2),'all'), mean(im(:,:,3),'all')];
end
fill_color = cast(fill_color, 'like', im);
im_sz = size(im, [1 2]);
cx = get_center_m(original_sz(2));
cy = get_center_m(original_sz(1));

cxt_xmin = round(pos(2)-cx);
cxt_xmax = round(cxt_xmin+original_sz(2)-1);
cxt_ymin = round(pos(1)-cy);
cxt_ymax = round(cxt_ymin+original_sz(1)-1);
left_pad = round(max([0, -cxt_xmin]));
top_pad = round(max([0, -cxt_ymin]));
right_pad = round(max([0, cxt_xmax-im_sz(2)+1]));
bottom_pad = round(max([0, cxt_ymax-im_sz(1)+1]));

cxt_xmin = cxt_xmin + left_pad;
cxt_xmax = cxt_xmax + left_pad;
cxt_ymin = cxt_ymin + top_pad;
cxt_ymax = cxt_ymax + top_pad;
if top_pad > 0 || bottom_pad > 0 || left_pad > 0 || right_pad > 0
   im_pad = repmat(reshape(fill_color,1,1,3),top_pad+im_sz(1)+bottom_pad,left_pad+im_sz(2)+right_pad,1);
   im_pad(top_pad+1:top_pad+im_sz(1), left_pad+1:left_pad+im_sz(2), :) = im;
else
    im_pad = im;
end

im_patch = im_pad(cxt_ymin+1:cxt_ymax+1, cxt_xmin+1:cxt_xmax+1, :);
if size(im_patch,1)~=model_sz(1) || size(im_patch,2)~=model_sz(2)
    im_patch = myMexResize(im_patch,model_sz,'lanczos');
end

end

