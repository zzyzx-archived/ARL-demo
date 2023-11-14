function [rec] = myMinRec(box1,box2)
%MYMINREC Summary of this function goes here
%   Detailed explanation goes here
xmin = min([box1(1) box2(1)]);
ymin = min([box1(2) box2(2)]);
xmax = max([box1(1)+box1(3)-1 box2(1)+box2(3)-1]);
ymax = max([box1(2)+box1(4)-1 box2(2)+box2(4)-1]);
rec = [xmin ymin xmax-xmin+1 ymax-ymin+1];
end

