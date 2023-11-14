function [predx,predy] = linearfit(pos,window,mode)
%LINEARFIT Summary of this function goes here
%   Detailed explanation goes here

len = min(window,size(pos,1));
if mode=='xy'
    data = pos(end-len+2:end,1:2)-pos(end-len+1:end-1,1:2);
else
    data = pos(end-len+2:end,3:4)-pos(end-len+1:end-1,3:4);
end

[U,mu,vars] = pca(data');
if isempty(U)
    predx = mu(1);
    predy = mu(2);
else
    [coe,~,~] = pcaApply(data',U,mu,1);
    P = polyfit(1:(len-1),coe(1,:),1);
    pred_coe1 = polyval(P,len+1);
    pred = pred_coe1*U(:,1)+mu;
    predx = pred(1);
    predy = pred(2);
end

end

