function [score] = getScoreInBox_m(mat,bbox)
%GETSCOREINBOX_M Summary of this function goes here
%   Detailed explanation goes here

score = 0;
if ~isempty(bbox)
    [m,n] = size(mat,[1 2]);
    xmin = round(bbox(1));
    xmax = round(bbox(1)+bbox(3)-1);
    xmin = max(1,xmin);
    xmax = min(n,xmax);
    ymin = round(bbox(2));
    ymax = round(bbox(2)+bbox(4)-1);
    ymin = max(1,ymin);
    ymax = min(m,ymax);
    tmp = mat(ymin:ymax, xmin:xmax);
    if ~isempty(tmp)
        score = mean(tmp,'all');
    end
end

end

