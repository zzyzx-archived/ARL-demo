function [feature_map,par_saab] = get_saab( im, fparams, gparams )
%GET_SAAB Summary of this function goes here
%   Detailed explanation goes here
[im_height, im_width, num_im_chan, num_images] = size(im);
single_im = single(im)/255;
par_saab = {};

if strcmp(fparams.colorspace,'gray')
    if num_im_chan == 3
        if num_images == 1
            t_colorspace = rgb2gray(single_im) - 0.5;
        else
            t_colorspace = zeros(im_height, im_width, 1, num_images, 'single');
            for k = 1:num_images
                t_colorspace(:,:,:,k) = rgb2gray(single_im(:,:,:,k)) - 0.5;
            end
        end
    elseif num_im_chan == 1
        t_colorspace = single_im - 0.5;
    else
        except = MException('get_saab','Invalid input data, must have 1 or 3 dimensions');
        throw(except);
    end
elseif strcmp(fparams.colorspace,'rgb')
    if num_im_chan == 3
        t_colorspace = single_im - 0.5;
    else
        except = MException('get_saab','Invalid input data, must have 3 dimensions for rgb');
        throw(except);
    end
end

pad_im = t_colorspace;

% train or test
is_train = fparams.is_train;
if is_train
    if num_im_chan==1
        if num_images==1
            batch = im2col(pad_im, [fparams.ksize fparams.ksize]);
            batch = batch(:,1:fparams.stride:end); % [fdim num_sample]
        else
            for k=1:num_images
                tmp = im2col(pad_im(:,:,:,k), [fparams.ksize fparams.ksize]);
                batch{k} = tmp(:,1:fparams.stride:end);
            end
            batch = cat(2,batch{:});
        end
    elseif num_im_chan==3
        if num_images==1
            for k=1:num_im_chan
                tmp = im2col(pad_im(:,:,k), [fparams.ksize fparams.ksize]);
                batch{k} = tmp(:,1:fparams.stride:end); % [fdim num_sample]
            end
            batch = cat(1,batch{:});
        else
            for k=1:num_images
                for p=1:num_im_chan
                    tmp = im2col(pad_im(:,:,p,k), [fparams.ksize fparams.ksize]);
                    batch_s{p} = tmp(:,1:fparams.stride:end);
                end
                batch{k} = cat(1,batch_s{:});
            end
            batch = cat(2,batch{:});
        end
    else
        except = MException('get_saab','Invalid input data for training');
        throw(except);
    end
    batch = batch-mean(batch,2);% sample mean
    batch = batch-mean(batch,1);% dc
    [kernels,mu,vars] = pca(batch);
    kernels = reshape(kernels,fparams.ksize,fparams.ksize,[],size(kernels,2));
    kernels = kernels(:,:,:,1:end-1);
    dc_kernel = ones(size(kernels,[1 2 3]));
    dc_kernel = dc_kernel./(norm(dc_kernel(:)));
    par_saab.dc_kernel = dc_kernel;
    par_saab.kernels = kernels;
    par_saab.train_mean = mu;
    par_saab.train_vars = vars;
    
else
    par_saab = fparams.par_saab;
    dc_kernel = par_saab.dc_kernel;
    kernels = par_saab.kernels;
    %norm_min = par_saab.min;
    %norm_max = par_saab.max;
    if size(kernels,4)>(fparams.num_kernel-1)
        kernels = kernels(:,:,:,1:(fparams.num_kernel-1));
        %norm_min = norm_min(:,:,1:fparams.num_kernel);
        %norm_max = norm_max(:,:,1:fparams.num_kernel);
    end
    if fparams.feature_weight
        weight = sqrt(par_saab.train_vars(1:size(kernels,4)));
        weight = weight./sum(weight);
        weight = weight./weight(1);
        weight = [1;weight];
        weight = reshape(weight,1,1,[]);
    end
end

% pad the image to maintain same spatial size
if fparams.do_pad
    pad_num = fparams.pad_num;
    pad_im = padarray(pad_im,[pad_num pad_num],'symmetric','both');
end

% convolution with kernels
feature_map = cell(1+size(kernels,4),1);
feature_map{1} = convn(pad_im,dc_kernel,'valid');
if fparams.stride>1
    feature_map{1} = feature_map{1}(1:fparams.stride:end,1:fparams.stride:end,:,:);
end
for k=1:size(kernels,4)
    feature_map{k+1} = convn(pad_im,kernels(:,:,:,k),'valid');
    if fparams.stride>1
        feature_map{k+1} = feature_map{k+1}(1:fparams.stride:end,1:fparams.stride:end,:,:);
    end
end
feature_map = cat(3,feature_map{:});

% normalization
% if is_train
%     % normalization stats
%     delta = 1e-7;
%     par_saab.min = min(feature_map,[],[1 2 4]);
%     par_saab.max = max(feature_map,[],[1 2 4])+delta;
%     
%     norm_min = par_saab.min;
%     norm_max = par_saab.max;
% end


if gparams.cell_size > 1
    if fparams.maxpool>1
        [feature_map,~] = MaxPooling(feature_map,[single(fparams.maxpool),single(fparams.maxpool)]);
    end
    if fparams.conca>1
        feature_map = myConca(feature_map, fparams.conca);
    end
    tmp = round(gparams.cell_size/(fparams.stride*fparams.maxpool*fparams.conca));
    if tmp>1
        feature_map = average_feature_region(feature_map,tmp);
    end
end

delta = 1e-7;
norm_min = min(feature_map,[],[1 2 4]);
norm_max = max(feature_map,[],[1 2 4])+delta;
feature_map = (feature_map-norm_min)./(norm_max-norm_min);

if ~is_train && fparams.feature_weight
    feature_map = feature_map.*weight;
end

end

