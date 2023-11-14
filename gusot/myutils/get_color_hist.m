function [vec] = get_color_hist(image, pos, sz)
%GET_COLOR_HIST Summary of this function goes here
%   Detailed explanation goes here
if max(sz)>200
    scale = 200/max(sz);
    img_sample = get_pixels(image, pos, round(sz),round(sz*scale));
else
    img_sample = get_pixels(image, pos, round(sz),[]);
end

params.tablename = 'CNnorm';
params.useForGray = false;
params.cell_size = 1;
params.nDim = 10;
gparams.use_gpu = false;
img_feat = get_table_feature(img_sample,params,gparams);
[~,ind] = max(reshape(img_feat,[prod(size(img_feat,[1 2])), size(img_feat,3)]), [], 2);
vec = hist(ind,1:params.nDim);
vec = vec/sum(vec);
if false
    figure;
    subplot(3,5,1),imshow(img_sample);
    for i=1:size(img_feat,3)
        subplot(3,5,5+i),imshow(img_feat(:,:,i));
    end
end
end

