function [meanColor] = myMeanColor(im,pos,sz)
%MYMEANCOLOR Summary of this function goes here
%   [cy,cx], [h,w]
m = size(im,1);
n = size(im,2);
upleft = pos-(sz-1)/2;
botright = pos+(sz-1)/2;
ymin = round(min([max([1 upleft(1)]) m]));
ymax = round(max([min([m botright(1)]) 1]));
xmin = round(min([max([1 upleft(2)]) n]));
xmax = round(max([min([n botright(2)]) 1]));
meanColor = mean(im(ymin:ymax,xmin:xmax,:),[1 2]);
end

