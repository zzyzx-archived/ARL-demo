function [bayes_seg,final_seg] = mrf(im,fg_coors,bg_coors,n_compo,max_iter,lmbd,lmbd2,verbose)
%MRF Summary of this function goes here
%   fg_coors,bg_coors are 1*2 cell, sub indexes {i,j} = ind2sub(size(a),find(a>0.5))
if nargin<8
    verbose = false;
end
[height, width] = size(im,[1 2]);
bayes_seg = zeros(height,width);
final_seg = zeros(height,width);

fg_pixels = index_3darray(im,fg_coors{1},fg_coors{2});
bg_pixels = index_3darray(im,bg_coors{1},bg_coors{2});
fg_gmm = gmm_fit(fg_pixels,n_compo);
bg_gmm = gmm_fit(bg_pixels,n_compo);

ks = -ones(height, width);
ks_fg_prior = gmm_predict(fg_gmm,fg_pixels);
ks_bg_prior = gmm_predict(bg_gmm,bg_pixels);
ks(sub2ind(size(ks), fg_coors{1}, fg_coors{2})) = ks_fg_prior;
ks(sub2ind(size(ks), bg_coors{1}, bg_coors{2})) = ks_bg_prior;

alphas = -ones(height,width);
alphas(sub2ind(size(alphas), fg_coors{1}, fg_coors{2})) = 0;
alphas(sub2ind(size(alphas), bg_coors{1}, bg_coors{2})) = 1;

[uninit_coors{1},uninit_coors{2}] = ind2sub(size(alphas),find(alphas == -1));
uninit = index_3darray(im,uninit_coors{1},uninit_coors{2});

ks_fg = gmm_predict(fg_gmm,uninit);
ds_fg = gmm_score_samples(fg_gmm,uninit);
ks_bg = gmm_predict(bg_gmm,uninit);
ds_bg = gmm_score_samples(bg_gmm,uninit);
ks_fg_bg = [ks_fg, ks_bg];
fg_or_bg = ones(size(uninit,1),1);
fg_or_bg(ds_fg<ds_bg) = 0;

[fgLogPL,bgLogPL] = calc_NLL_gmm(fg_pixels,bg_pixels,uninit,n_compo);
fg_or_bg = ones(size(uninit,1),1);
fg_or_bg(fgLogPL<bgLogPL) = 0;

alphas(alphas == -1) = fg_or_bg;
for i=1:length(fg_or_bg)
    ks(uninit_coors{1}(i), uninit_coors{2}(i)) = ks_fg_bg(i, fg_or_bg(i)+1);
end

bayes_seg = ones(height,width) - alphas;

ws_hor = zeros(height, width);
ws_ver = zeros(height, width);
temp = 0;
count = 0;
for i=1:height
    for j=1:width
        if i < height
            temp = temp + sum((im(i, j, :) - im(i + 1, j, :)) .^ 2);
            count = count + 1;
        end
        if j < width
            temp = temp + sum((im(i, j, :) - im(i, j + 1, :)) .^ 2);
            count = count + 1;
        end
    end
end
beta = (2 * temp / count) .^ (-1);
ws_hor(:, 1:end-1) = (exp(-beta * sum((im(:, 2:end, :) - im(:, 1:end-1, :)) .^ 2, 3)) + lmbd2) * lmbd;
ws_ver(1:end-1, :) = (exp(-beta * sum((im(2:end, :,:) - im(1:end-1, :,:)) .^ 2, 3)) + lmbd2) * lmbd;

sij = ones(2,2) - eye(2);

[fg_gmm, bg_gmm] = update_gmms({fg_gmm, bg_gmm}, im, ks, alphas);

data_loss = zeros(height, width, 2);
for i=1:length(fg_coors)
    %data_loss[fg_coors + (np.zeros(len(fg_coors_zipped)).astype(np.int),)] = 0;
    data_loss(fg_coors{1}(i),fg_coors{2}(i),2) = Inf;
end
for i=1:length(bg_coors)
    data_loss(bg_coors{1}(i),bg_coors{2}(i),1) = Inf;
    %data_loss[bg_coors + (np.ones(len(bg_coors_zipped)).astype(np.int),)] = 0;
end

data_loss = update_data_loss(data_loss, uninit, uninit_coors{1}, uninit_coors{2}, {fg_gmm, bg_gmm});

flag_rows = ones(height,1);
flag_cols = ones(width,1);

old_alphas = alphas;

alphas = update_rows(alphas, data_loss, ws_hor, ws_ver, flag_rows, sij, height, width);
alphas = update_cols(alphas, data_loss, ws_hor, ws_ver, flag_cols, sij, height, width);

[new_de, new_se] = energy(alphas, data_loss, ws_hor, ws_ver, sij, height, width);
new_energy = new_de + new_se;
if verbose
    fprintf('Iteration 1 --- Energy: %f (DE=%f, SE=%f)\n',new_energy, new_de, new_se);
end

for i=2:max_iter
    old_energy = new_energy;

    old_flag_rows = flag_rows;
    old_flag_cols = flag_cols;
    flag_rows(:) = false;
    flag_cols(:) = false;
    for j=1:height
        if old_flag_rows(j)
            if any(old_alphas(j,:) ~= alphas(j,:))
                if j > 1
                    flag_rows(j - 1) = true;
                end
                if j < height
                    flag_rows(j + 1) = true;
                end
            end
        end
    end
    for j=1:width
        if old_flag_cols(j)
            if any(old_alphas(:, j) ~= alphas(:, j))
                if j > 1
                    flag_cols(j - 1) = true;
                end
                if j < width
                    flag_cols(j + 1) = true;
                end
            end
        end
    end
    
    ks = update_ks(ks, im, {fg_gmm, bg_gmm}, alphas);
    [fg_gmm, bg_gmm] = update_gmms({fg_gmm, bg_gmm}, im, ks, alphas);
    data_loss = update_data_loss(data_loss, uninit, uninit_coors{1}, uninit_coors{2}, {fg_gmm, bg_gmm});
    
    old_alphas = alphas;

    alphas = update_rows(alphas, data_loss, ws_hor, ws_ver, flag_rows, sij, height, width);
    alphas = update_cols(alphas, data_loss, ws_hor, ws_ver, flag_cols, sij, height, width);

    [new_de, new_se] = energy(alphas, data_loss, ws_hor, ws_ver, sij, height, width);
    new_energy = new_de + new_se;
    if verbose
        fprintf('Iteration %d --- Energy: %f (DE=%f, SE=%f)\n',i, new_energy, new_de, new_se);
    end

    if isnan(new_energy) || new_energy >= old_energy || new_energy== old_energy
        break;
    end
end

final_seg = ones(height,width) - alphas;

end

function [fgLogPL,bgLogPL] = calc_NLL_gmm(fgpixels,bgpixels,pixels,K)
[means_fg,covs_fg,priors_fg] = vl_gmm(fgpixels',K);
[means_bg,covs_bg,priors_bg] = vl_gmm(bgpixels',K);
allBGLogPL = zeros(size(pixels,1),K);
allFGLogPL = zeros(size(pixels,1),K);
for k=1:K
    %%% Get the k Gaussian weights for Background & Forground 
    bgGaussianWeight = priors_bg(k);
    fgGaussianWeight = priors_fg(k);

    %%% FOR ALL PIXELS - calculate the distance from the k gaussian (BG & FG)
    bgDist = pixels - repmat(means_bg(:,k)',size(pixels,1),1);
    fgDist = pixels - repmat(means_fg(:,k)',size(pixels,1),1);

    %%% Calculate the gaussian covariance matrix & use it to calculate
    %%% all of the pixels likelihood to it :
    bgCovarianceMat = diag(covs_bg(:,k));
    fgCovarianceMat = diag(covs_fg(:,k));
    allBGLogPL(:,k) = -log(bgGaussianWeight)+0.5*log(det(bgCovarianceMat)) + 0.5*sum( (bgDist/bgCovarianceMat).*bgDist, 2 );
    allFGLogPL(:,k) = -log(fgGaussianWeight)+0.5*log(det(fgCovarianceMat)) + 0.5*sum( (fgDist/fgCovarianceMat).*fgDist, 2 );
end
%%% Last, as seen in the GrabCut paper, take the minimum Log likelihood
%%% (    argmin(Dn)    )
bgLogPL = min(allBGLogPL, [], 2);
fgLogPL = min(allFGLogPL, [], 2);

end

function out = update_data_loss(data_loss, pixels, row_coors, col_coors, K, alphas, im)
out = data_loss;
coors = find(alphas == 0);
[r,c] = ind2sub(size(alphas),coors);
fgpixels = index_3darray(im,r,c);
coors = find(alphas == 1);
[r,c] = ind2sub(size(alphas),coors);
bgpixels = index_3darray(im,r,c);
[fgLogPL,bgLogPL] = calc_NLL_gmm(fgpixels,bgpixels,pixels,K);
for j=1:size(pixels,1)
    out(row_coors(j), col_coors(j), 1) = fgLogPL(j);
    out(row_coors(j), col_coors(j), 2) = bgLogPL(j);
end


end

function out = update_ks(ks, im, gmms, alphas)
out = ks;
for i=1:2
    coors = find(alphas == i-1);
    [r,c] = ind2sub(size(alphas),coors);
    pxs = index_3darray(im,r,c);
    out(coors) = gmm_predict(gmms{i},pxs);
end

end

function [fg_gmm,bg_gmm] = update_gmms(gmms_ref, im, ks, alphas)
% gmms: 1*2 cell
gmms = cell(1,2);
for a=1:2
    n_compo = size(gmms_ref{a}.mu,1);
    cnt = 0;
    mu = zeros(n_compo,3);
    covmat = zeros(3,3,n_compo);
    pis = zeros(1, n_compo);
    for k=1:n_compo
        [r,c] = ind2sub(size(ks), find(ks == k & alphas == a-1));
        k_pixels = index_3darray(im,r,c);
        if size(k_pixels,1)>3
            cnt = cnt + 1;
            pis(cnt) = size(k_pixels,1);
            mu(cnt,:) = mean(k_pixels, 1);
            covmat(:,:,cnt) = cov(k_pixels);
            try
                chol(covmat(:,:,cnt));
            catch ME
                [];
            end
        end
    end
    
    mu = mu(1:cnt,:);
    covmat = covmat(:,:,1:cnt);
    pis = pis(1:cnt);
    
    pis = pis / sum(pis);
    try
        gmms{a} = gmdistribution(mu,covmat,pis);
    catch ME
        [];
    end
end
fg_gmm = gmms{1};
bg_gmm = gmms{2};

end

function out_alphas = update_rows(alphas, data_loss, ws_hor, ws_ver, flag, s, h, w)
out_alphas = alphas;
for i=1:h
    if flag(i)
        ds_temp = zeros(2, w);
        pre = zeros(2, w);
        ds_temp(1, :) = ds_temp(1, :) + data_loss(i, :, 1);
        ds_temp(2, :) = ds_temp(2, :) + data_loss(i, :, 2);
        if i > 1
            ds_temp(1, :) = ds_temp(1, :) + ws_hor(i - 1,:) .* index_2darray(s,1+alphas(i - 1, :),1+zeros(1,w));
            ds_temp(2, :) = ds_temp(2, :) + ws_hor(i - 1,:) .* index_2darray(s,1+alphas(i - 1, :),1+ones(1,w));
        end
        if i < h
            ds_temp(1, :) = ds_temp(1, :) + ws_hor(i, :) .* index_2darray(s,1+alphas(i + 1, :),1+zeros(1,w));
            ds_temp(2, :) = ds_temp(2, :) + ws_hor(i, :) .* index_2darray(s,1+alphas(i + 1, :),1+ones(1,w));
        end
        for j=2:w
            if ds_temp(1, j - 1) < ds_temp(2, j - 1)
                ds_temp(1, j) = ds_temp(1, j) + ds_temp(1, j - 1);
                pre(1, j) = 0;
                smth = ds_temp(1, j - 1) + ws_ver(i, j - 1);
                if ds_temp(2, j - 1) < smth
                    ds_temp(2, j) = ds_temp(2, j) + ds_temp(2, j - 1);
                    pre(2, j) = 1;
                else
                    ds_temp(2, j) = ds_temp(2, j) + smth;
                    pre(2, j) = 0;
                end
            else
                ds_temp(2, j) = ds_temp(2, j) + ds_temp(2, j - 1);
                pre(2, j) = 1;
                smth = ds_temp(2, j - 1) + ws_ver(i, j - 1);
                if ds_temp(1, j - 1) < smth
                    ds_temp(1, j) = ds_temp(1, j) + ds_temp(1, j - 1);
                    pre(1, j) = 0;
                else
                    ds_temp(1, j) = ds_temp(1, j) + smth;
                    pre(1, j) = 1;
                end
            end
        end
        new_alphas = -ones(1,w);
        if ds_temp(1, end) < ds_temp(2, end)
            new_alphas(end) = 0;
        else
            new_alphas(end) = 1;
        end
        for row=2:w
            new_alphas(end-row+1) = pre(1+new_alphas(end-row+2), end-row+2);
        end
        out_alphas(i,:) = new_alphas;
    end
end

end

function out_alphas = update_cols(alphas, data_loss, ws_hor, ws_ver, flag, s, h, w)
out_alphas = alphas;
for i=1:w
    if flag(i)
        ds_temp = zeros(h, 2);
        pre = zeros(h, 2);
        ds_temp(:, 1) = ds_temp(:, 1) + data_loss(:, i, 1);
        ds_temp(:, 2) = ds_temp(:, 2) + data_loss(:, i, 2);
        if i > 1
            ds_temp(:, 1) = ds_temp(:, 1) + ws_hor(:, i - 1) .* index_2darray(s,1+alphas(:, i - 1),1+zeros(h,1));
            ds_temp(:, 2) = ds_temp(:, 2) + ws_hor(:, i - 1) .* index_2darray(s,1+alphas(:, i - 1),1+ones(h,1));
        end
        if i < w
            ds_temp(:, 1) = ds_temp(:, 1) + ws_hor(:, i) .* index_2darray(s,1+alphas(:, i + 1),1+zeros(h,1));
            ds_temp(:, 2) = ds_temp(:, 2) + ws_hor(:, i) .* index_2darray(s,1+alphas(:, i + 1),1+ones(h,1));
        end
        for j=2:h
            if ds_temp(j - 1, 1) < ds_temp(j - 1, 2)
                ds_temp(j, 1) = ds_temp(j, 1) + ds_temp(j - 1, 1);
                pre(j, 1) = 0;
                smth = ds_temp(j - 1, 1) + ws_ver(j - 1, i);
                if ds_temp(j - 1, 2) < smth
                    ds_temp(j, 2) = ds_temp(j, 2) + ds_temp(j - 1, 2);
                    pre(j, 2) = 1;
                else
                    ds_temp(j, 2) = ds_temp(j, 2) + smth;
                    pre(j, 2) = 0;
                end
            else
                ds_temp(j, 2) = ds_temp(j, 2) + ds_temp(j - 1, 2);
                pre(j, 2) = 1;
                smth = ds_temp(j - 1, 2) + ws_ver(j - 1,i);
                if ds_temp(j - 1, 1) < smth
                    ds_temp(j, 1) = ds_temp(j, 1) + ds_temp(j - 1, 1);
                    pre(j, 1) = 0;
                else
                    ds_temp(j, 1) = ds_temp(j, 1) + smth;
                    pre(j, 1) = 1;
                end
            end
        end
        new_alphas = -ones(h,1);
        if ds_temp(end, 1) < ds_temp(end, 2)
            new_alphas(end) = 0;
        else
            new_alphas(end) = 1;
        end
        for row=2:h
            new_alphas(end-row+1) = pre(end-row+2, 1+new_alphas(end-row+2));
        end
        out_alphas(:, i) = new_alphas;
    end
end
        
end

function [de,se] = energy(alphas, data_loss, ws_hor, ws_ver, s, h, w)
de = 0;
se = 0;
for i=1:h
    for j=1:w
        de = de + data_loss(i, j, 1+alphas(i, j));
        if i < h - 1
            se = se + s(1+alphas(i, j), 1+alphas(i + 1, j)) * ws_ver(i, j);
        end
        if j < w - 1
            se = se + s(1+alphas(i, j), 1+alphas(i, j + 1)) * ws_hor(i, j);
        end
    end
end

end

