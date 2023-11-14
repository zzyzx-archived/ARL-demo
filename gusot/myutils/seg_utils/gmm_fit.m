function gmm = gmm_fit(X,n_comp)
%GMM_FIT Summary of this function goes here
%   Detailed explanation goes here
rng(0); % For reproducibility

% reset warnings
% lastwarn('');

try
    gmm = fitgmdist(X,n_comp,'RegularizationValue',1e-6);
catch ME
    if strcmp(ME.identifier,'stats:gmdistribution:wdensity:IllCondCov')
        gmm = fitgmdist(X,1);
    end
end

% Check which warning occured (if any)
% [msgstr, msgid] = lastwarn;
% if strcmp(msgid,'stats:gmdistribution:FailedToConverge')
%     gmm = fitgmdist(X,1);
% end

end

