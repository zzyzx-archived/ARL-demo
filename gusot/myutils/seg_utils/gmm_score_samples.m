function [nlogs] = gmm_score_samples(gmm,X)
%GMM_SCORE_SAMPLES Summary of this function goes here
%   Detailed explanation goes here
nlogs = zeros(size(X,1),1);
for i=1:size(X,1)
    [~,nlogs(i)] = posterior(gmm,X(i,:));
end

end

