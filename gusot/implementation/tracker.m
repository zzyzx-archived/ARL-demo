function [results,my] = tracker(params)
import py.heatmap_detection.*
import py.segment.*
%import py.box_regression.*

warning('off','stats:gmdistribution:FailedToConverge');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get sequence info
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return;
end
results.fps = 0;

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

% my
my.seg_box = zeros(seq.len,4);
my.gmc_box = zeros(seq.len,4);
my.gmc_box(1,:) = [seq.init_pos(2)-(seq.init_sz(2)-1)/2, seq.init_pos(1)-(seq.init_sz(1)-1)/2, seq.init_sz(2), seq.init_sz(1)];
my.traj_box = zeros(seq.len,4);
my.score = zeros(seq.len,4);
my.gmc_valid = zeros(seq.len);
my.motion = zeros(seq.len,5);
my.output_box = zeros(seq.len,4);
my.output_box_ori = zeros(seq.len,4);
my.pos_sz = zeros(seq.len,4);
my.chose = ones(seq.len,1);
last_im_gray = im2gray(im);
if max(size(im))>1000
    if prod(seq.init_sz)>6000
        dif_ds_stride = 3;
    else
        dif_ds_stride = 2;
    end
    last_im_gray = last_im_gray(1:dif_ds_stride:end,1:dif_ds_stride:end);
else
    dif_ds_stride = 1;
end
last_im_gray = py.numpy.array(last_im_gray);
hit_obj = false;
flag_lv = false;
my.flag_lv_num = 0;
my.no_gmc_cnt = -1;
flag_large = (seq.init_sz(1)/size(im,1)>0.5 && seq.init_sz(2)/size(im,2)>0.5);
my.rapid_change = zeros(seq.len,1);
flag_small = prod(seq.init_sz)<1000;
if flag_large
    disp('large obj');
elseif flag_small
    disp('small obj');
end
flag_seg = true;
flag_seg_sp = false;
numGoodIouMotion = 0;

% timer
my.timer = zeros(seq.len,2);
my.update_rate = zeros(seq.len,1);
if is_color_image
    my.meanColor = zeros(seq.len,3);
    my.meanColor_trust = zeros(seq.len,3);
else
    my.meanColor = zeros(seq.len,1);
    my.meanColor_trust = zeros(seq.len,1);
end
my.occ = zeros(seq.len,1);

% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end

global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;

% Load learning parameters
admm_max_iterations = params.max_iterations;
init_penalty_factor = params.init_penalty_factor;
max_penalty_factor = params.max_penalty_factor;
penalty_scale_step = params.penalty_scale_step;
temporal_regularization_factor = params.temporal_regularization_factor; 

init_target_sz = target_sz;

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor(base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2];
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'exact');

% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = unique(feature_info.data_sz, 'rows', 'stable');
feature_cell_sz = unique(feature_info.min_cell_size, 'rows', 'stable');
num_feature_blocks = size(feature_sz, 1);

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);

% Size of the extracted feature maps
feature_sz_cell = mat2cell(feature_sz, ones(1,num_feature_blocks), 2);
filter_sz = feature_sz;
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size
[output_sz, k1] = max(filter_sz, [], 1);
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% Construct the Gaussian label function
yf = cell(numel(num_feature_blocks), 1);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma = sqrt(prod(floor(base_target_sz/feature_cell_sz(i)))) * params.output_sigma_factor;
    rg           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]     = ndgrid(rg,cg);
    y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
    yf{i}           = fft2(y); 
end

% Compute the cosine windows
cos_window = cellfun(@(sz) hann(sz(1))*hann(sz(2))', feature_sz_cell, 'uniformoutput', false);

% Define spatial regularization windows
reg_window = cell(num_feature_blocks, 1);
for i = 1:num_feature_blocks
    reg_scale = floor(base_target_sz/params.feature_downsample_ratio(i));
    use_sz = filter_sz_cell{i};    
    reg_window{i} = ones(use_sz) * params.reg_window_max;
    range = zeros(numel(reg_scale), 2);
    
    % determine the target center and range in the regularization windows
    for j = 1:numel(reg_scale)
        range(j,:) = [0, reg_scale(j) - 1] - floor(reg_scale(j) / 2);
    end
    center = floor((use_sz + 1)/ 2) + mod(use_sz + 1,2);
    range_h = (center(1)+ range(1,1)) : (center(1) + range(1,2));
    range_w = (center(2)+ range(2,1)) : (center(2) + range(2,2));
    
    reg_window{i}(range_h, range_w) = params.reg_window_min;
end

% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;

% Use the translation filter to estimate the scale
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
scaleFactors = scale_step .^ scale_exp;

if nScales > 0
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

seq.time = 0;

% Define the learning variables
f_pre_f = cell(num_feature_blocks, 1);
cf_f = cell(num_feature_blocks, 1);

% Allocate
scores_fs_feat = cell(1,1,num_feature_blocks);
while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        curr_im_gray = im2gray(im);
        if dif_ds_stride>1
            curr_im_gray = curr_im_gray(1:dif_ds_stride:end,1:dif_ds_stride:end);
        end
        curr_im_gray = py.numpy.array(curr_im_gray);
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end

    tic();
    
    if seq.frame==11 && ~(flag_seg || flag_seg_small)
        if target_sz(1)<4*target_sz(2) && target_sz(1)>0.25*target_sz(2)
            flag_seg_sp = true;
        end
    end
    if flag_seg_sp && seq.frame==401 && (sum(my.score(1:seq.frame,2)>0.15)>0.5*seq.frame || numGoodIouMotion<5)
        flag_seg_sp = false;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% seg
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if seq.frame==1 && is_color_image==true
        seg_mode = 'pdf';
        res = py.segment.init_seg_params(seq.image_files{seq.frame},py.list(seq.init_rect),seg_mode);
        %res=cell(res);
        flag_seg = logical(res{1});
        flag_seg_small = false;
        if flag_seg
            display('use pre-defined color keys');
            %results.res = [];
            %return;
        end
        try
            if ~flag_seg
                display('try adaptive color keys');
                tmp = double(res{3});
                tmp = reshape(tmp,prod(size(tmp,[1 2])),[]);
                tmp = gmm_fit(tmp,5);
                if tmp.NumComponents>1
                    res = py.segment.init_seg_params(seq.image_files{seq.frame},py.list(seq.init_rect),'self',tmp.mu);
                    flag_seg = logical(res{1});
                end
            end
        catch
            [];
        end

        if target_sz(1)<=50 && target_sz(2)<=50
            flag_seg = false;
            flag_seg_small = true;
        end
        init_colorkey_info = res{2};
        fprintf('init box size (%d,%d), %d\n', target_sz(1), target_sz(2), prod(target_sz)>200*200);
    end
    if is_color_image==false
        flag_seg = false;
        flag_seg_small = false;
        flag_seg_sp = false;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% GMC
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    flag_detect = true;
    flag_update = true;
    force_update = false;
    strong_update = false;
    gmc_success = false;
    gmc_pos = [];
    if seq.frame>1
        last_pred = seq.rect_position(seq.frame-1,:);
        flag_small = prod(last_pred([3 4]))<1000;
    end
    if seq.frame>1 && ~flag_lv
        w = int32(last_pred(3));
        h = int32(last_pred(4));
        
        if flag_small && my.motion(seq.frame-1,1)>1
            hit_obj_radius = min([max([max(last_pred([3,4])), 30]), 50]);
            res = py.heatmap_detection.GMC_total_all(last_im_gray,curr_im_gray,...
                        [int32(h/dif_ds_stride), int32(w/dif_ds_stride)],...
                        int32(1),0.2,0,...
                        int32((my.gmc_box(seq.frame-1,2)-1)/dif_ds_stride),...
                        int32((my.gmc_box(seq.frame-1,1)-1)/dif_ds_stride),...
                        int32(hit_obj_radius));
        else
            res = py.heatmap_detection.GMC_total_all(last_im_gray,curr_im_gray,...
                        [int32(h/dif_ds_stride), int32(w/dif_ds_stride)],...
                        int32(1),0.2,0.1,...
                        int32((my.gmc_box(seq.frame-1,2)-1)/dif_ds_stride),...
                        int32((my.gmc_box(seq.frame-1,1)-1)/dif_ds_stride),...
                        int32(-1));
        end
        
        res=cell(res);
        gmc_success=logical(res{1});
        dif_map = double(res{4});
        dif_map_py = res{4};
        
        if ~gmc_success
            disp('large lv');
            flag_lv = true;
            my.flag_lv_num = my.flag_lv_num+1;
            my.no_gmc_cnt = 0;
            gmc_pos = [];
        else
            gmc_pos_ori = double(res{3});
            dif_int = double(res{2});
            if ~isempty(gmc_pos_ori)
                gmc_pos = dif_ds_stride*gmc_pos_ori+1;% dif of index in python and matlab
                my.gmc_box(seq.frame,:) = [gmc_pos(2), gmc_pos(1), w, h];
                my.motion(seq.frame,1) = dif_int(int32(gmc_pos_ori(1)+1), int32(gmc_pos_ori(2)+1));
            else
                gmc_pos = [];
                my.gmc_box(seq.frame,:) = my.gmc_box(seq.frame-1,:);
            end
        end
    end
    
    dx_lf = [];
    dy_lf = [];
    if seq.frame>10
        [dx_lf,dy_lf] = linearfit(seq.rect_position(1:seq.frame-1,:),20,'xy');
        tmp = seq.rect_position(seq.frame-1,:);
        my.traj_box(seq.frame,:) = [tmp(1)+dx_lf, tmp(2)+dy_lf, w, h];
        if gmc_success
            [my.motion(seq.frame,2),~] = getValue_my(dif_int, int32((tmp(2)+dy_lf)/dif_ds_stride), int32((tmp(1)+dx_lf)/dif_ds_stride));
        end
    end
    
    if isfield(seq,'gt_boxes')
        gt_boxes = seq.gt_boxes(seq.frame,:);
        if gmc_success
            [my.motion(seq.frame,4),~] = getValue_my(dif_int, int32(gt_boxes(2)/dif_ds_stride), int32(gt_boxes(1)/dif_ds_stride));
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we 
    % just want to initialize the tracker there
    if seq.frame > 1
        if gmc_success
            [motion_old_pos,out_of_dif_old] = getValue_my(dif_int, (pos(1)-(last_pred(4)-1)/2)/dif_ds_stride, (pos(2)-(last_pred(3)-1)/2)/dif_ds_stride);
        end
        last_pos = pos;
        if flag_detect
            old_pos = inf(size(pos));
            iter = 1;

            % gmc
            if ~isempty(gmc_pos)
                gmc_ct = round([gmc_pos(1)+(last_pred(4)-1)/2, gmc_pos(2)+(last_pred(3)-1)/2]);
                [xt_gmc,~] = extract_features(im, gmc_ct, currentScaleFactor, features, global_fparams, feature_extract_info);
                xtw_gmc = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt_gmc, cos_window, 'uniformoutput', false);
                xtf_gmc = cellfun(@fft2, xtw_gmc, 'uniformoutput', false);
                scores_fs_feat_gmc{k1} = gather(sum(bsxfun(@times, conj(last_op_model{k1}), xtf_gmc{k1}), 3));
                scores_fs_sum_gmc = scores_fs_feat_gmc{k1};
                for k = block_inds
                    scores_fs_feat_gmc{k} = gather(sum(bsxfun(@times, conj(last_op_model{k}), xtf_gmc{k}), 3));
                    scores_fs_feat_gmc{k} = resizeDFT2(scores_fs_feat_gmc{k}, output_sz);
                    scores_fs_sum_gmc = scores_fs_sum_gmc +  scores_fs_feat_gmc{k};
                end
                scores_fs_gmc = gather(scores_fs_sum_gmc);
                responsef_padded_gmc = resizeDFT2(scores_fs_gmc, output_sz);
                response_gmc = ifft2(responsef_padded_gmc, 'symmetric');
                my.score(seq.frame,1) = response_gmc(1,1);
                %my.score(seq.frame,1) = max(response_gmc(:));
                
                scores_fs_feat_gmc_new{k1} = gather(sum(bsxfun(@times, conj(cf_f{k1}), xtf_gmc{k1}), 3));
                scores_fs_sum_gmc_new = scores_fs_feat_gmc_new{k1};
                for k = block_inds
                    scores_fs_feat_gmc_new{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xtf_gmc{k}), 3));
                    scores_fs_feat_gmc_new{k} = resizeDFT2(scores_fs_feat_gmc_new{k}, output_sz);
                    scores_fs_sum_gmc_new = scores_fs_sum_gmc_new +  scores_fs_feat_gmc_new{k};
                end
                scores_fs_gmc_new = gather(scores_fs_sum_gmc_new);
                responsef_padded_gmc_new = resizeDFT2(scores_fs_gmc_new, output_sz);
                response_gmc_new = ifft2(responsef_padded_gmc_new, 'symmetric');
                my.score(seq.frame,5) = response_gmc_new(1,1);
                %my.score(seq.frame,5) = response_gmc_new(1,1);
            end
            
            % traj
            if ~isempty(dx_lf)
                trj_ct = round(last_pos+[dy_lf,dx_lf]);
                [xt_trj,~] = extract_features(im, trj_ct, currentScaleFactor, features, global_fparams, feature_extract_info);
                xtw_trj = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt_trj, cos_window, 'uniformoutput', false);
                xtf_trj = cellfun(@fft2, xtw_trj, 'uniformoutput', false);
                scores_fs_feat_trj{k1} = gather(sum(bsxfun(@times, conj(cf_f{k1}), xtf_trj{k1}), 3));
                scores_fs_sum_trj = scores_fs_feat_trj{k1};
                for k = block_inds
                    scores_fs_feat_trj{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xtf_trj{k}), 3));
                    scores_fs_feat_trj{k} = resizeDFT2(scores_fs_feat_trj{k}, output_sz);
                    scores_fs_sum_trj = scores_fs_sum_trj +  scores_fs_feat_trj{k};
                end
                scores_fs_trj = gather(scores_fs_sum_trj);
                responsef_padded_trj = resizeDFT2(scores_fs_trj, output_sz);
                response_trj = ifft2(responsef_padded_trj, 'symmetric');
                my.score(seq.frame,3) = response_trj(1,1);
            end

            %translation search
            while iter <= params.refinement_iterations && any(old_pos ~= pos)
                % Extract features at multiple resolutions
                sample_pos = round(pos);
                sample_scale = currentScaleFactor*scaleFactors;
                [xt,~] = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info);

                % Do windowing of features
                xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);

                % Compute the fourier series
                xtf = cellfun(@fft2, xtw, 'uniformoutput', false);

                % Compute convolution for each feature block in the Fourier domain
                % and the sum over all blocks.
                scores_fs_feat{k1} = gather(sum(bsxfun(@times, conj(cf_f{k1}), xtf{k1}), 3));
                scores_fs_sum = scores_fs_feat{k1};
                for k = block_inds
                    scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xtf{k}), 3));
                    scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                    scores_fs_sum = scores_fs_sum +  scores_fs_feat{k};
                end

                % Also sum over all feature blocks.
                % Gives the fourier coefficients of the convolution response.
                scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);

                responsef_padded = resizeDFT2(scores_fs, output_sz);
                response = ifft2(responsef_padded, 'symmetric');
                [disp_row, disp_col, sind, max_scale_response] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, output_sz);
                my.score(seq.frame,2) = max_scale_response;        

                % Compute the translation vector in pixel-coordinates and round
                % to the closest integer pixel.
                translation_vec = [disp_row, disp_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(sind);            
                scale_change_factor = scaleFactors(sind);

                % update position
                old_pos = pos;
                pos = sample_pos + translation_vec;

                if params.clamp_position
                    pos = max([1 1], min([size(im,1) size(im,2)], pos));
                end

                % Update the scale
                currentScaleFactor_ori = currentScaleFactor;
                currentScaleFactor = currentScaleFactor * scale_change_factor;

                % Adjust to make sure we are not to large or to small
                if currentScaleFactor < min_scale_factor
                    currentScaleFactor = min_scale_factor;
                elseif currentScaleFactor > max_scale_factor
                    currentScaleFactor = max_scale_factor;
                end
                
                % suppress too small box
                tmp_sz = base_target_sz * currentScaleFactor;
                if tmp_sz(1)<10 || tmp_sz(2)<10
                    if tmp_sz(1)<10
                        currentScaleFactor = currentScaleFactor*(10/tmp_sz(1));
                    else
                        currentScaleFactor = currentScaleFactor*(10/tmp_sz(2));
                    end
                end

                iter = iter + 1;
            end
        end
    end
    pos_ori = pos;
    
    
    flag_occ = false;
    if seq.frame>1
        % baseline
        good1 = false;
        good2 = false;
        good3 = false;
    end
    if seq.frame>1 && gmc_success
        [motion_new_pos,out_of_dif_new] = getValue_my(dif_int, (pos(1)-(last_pred(4)-1)/2)/dif_ds_stride, (pos(2)-(last_pred(3)-1)/2)/dif_ds_stride);

        if motion_old_pos<0.1 && motion_new_pos<0.1 && ~out_of_dif_new && ~out_of_dif_old
            flag_update = false;
            my.gmc_valid(seq.frame) = 50;
        end
        
        % gmc
        good2 = my.motion(seq.frame,1)>5 && (my.score(seq.frame,1)>=my.score(seq.frame,2)...
            || (seq.frame>10 && (hit_obj||flag_large) && std(my.gmc_box(seq.frame-10:seq.frame,1))<40 && std(my.gmc_box(seq.frame-10:seq.frame,2))<40)...
            || (seq.frame>30 && motion_new_pos<1 && std(my.gmc_box(seq.frame-30:seq.frame,1))<20 && std(my.gmc_box(seq.frame-30:seq.frame,2))<20)...
            || (seq.frame>40 && std(my.gmc_box(1:seq.frame,1))<30 && std(my.gmc_box(1:seq.frame,2))<30) ...
            || (seq.frame>10 &&  (std(my.gmc_box(seq.frame-10:seq.frame,1))<70 && std(my.gmc_box(seq.frame-10:seq.frame,2))<20 || std(my.gmc_box(seq.frame-10:seq.frame,1))<20 && std(my.gmc_box(seq.frame-10:seq.frame,2))<70) )...
            );
        % trj
        good3 = calcRectInt(my.traj_box(seq.frame,:),my.gmc_box(seq.frame,:))>=0.5 && ...
                    my.score(seq.frame,3)>0.8*my.score(seq.frame,2) && norm([dy_lf,dx_lf])<(norm(translation_vec)/4);
    end
    
    if seq.frame>1
        flag_merge = false;% to merge multiple proposals
        
        good1 = my.score(seq.frame,2)>=0.08 && norm(translation_vec)<=min([30, 0.5*max(last_pred([3 4]))]);
        good3 = good3 || my.score(seq.frame,3)>=my.score(seq.frame,2) ||...
            my.score(seq.frame,3)>0.8*my.score(seq.frame,2) && norm([dy_lf,dx_lf])<(norm(translation_vec)/2);
        signal = [num2str(good1) num2str(good2) num2str(good3)];
        switch signal
            case '111'
                flag_merge = true;
            case '110'
                flag_merge = true;
                tmp_pos = [gmc_pos(1)+(last_pred(4)-1)/2, gmc_pos(2)+(last_pred(3)-1)/2];
                if norm(tmp_pos-pos)>max(last_pred([3 4]))
                    if my.score(seq.frame,1)>my.score(seq.frame,2)
                        tmp1 = myMeanColor(im,pos,base_target_sz * currentScaleFactor);
                        tmp1 = squeeze(tmp1)';
                        tmp2 = myMeanColor(im,tmp_pos,last_pred([4 3]));
                        tmp2 = squeeze(tmp2)';
                        tmp_dist1 = mean(abs(tmp1-my.meanColor_trust(1,:)),'all');
                        tmp_dist2 = mean(abs(tmp2-my.meanColor_trust(1,:)),'all');
                        if tmp_dist1>tmp_dist2+10
                            pos = tmp_pos;
                            currentScaleFactor = currentScaleFactor_ori;
                            flag_merge = false;
                        end
                    end
                end
            case '100'
                gmc_success;
            case '000'
                if prod(last_pred([3 4]))>2000 && is_color_image && mean(my.score(1:20,2))>0.1 && ~flag_large && (seq.frame>10 && mean(my.score(seq.frame-10:seq.frame,2))>0.06 && (my.score(seq.frame,2)-my.score(seq.frame-1,2)<0.01 || my.score(seq.frame,2)<0.05) ||...
                        (prod(last_pred([3 4]))>4500 && seq.frame>20 && mean(my.score(seq.frame-20:seq.frame,2))>0.08 && my.score(seq.frame,2)<0.08) )
                    tmp = myMeanColor(im,pos,base_target_sz * currentScaleFactor);
                    tmp = squeeze(tmp)';
                    tmp_dist = mean(abs(tmp-my.meanColor_trust(seq.frame-10,:)),'all');
                    if tmp_dist>20 || (tmp_dist>10 && my.score(seq.frame,2)-my.score(seq.frame-5,2)<-0.1)
                        flag_occ = true;
                        my.occ(seq.frame) = 1;
                        flag_update = false;
                        currentScaleFactor = currentScaleFactor_ori;
                    end
                end
                if seq.frame>20 && mean(my.score(1:20,2))>0.1 && ~flag_large && flag_occ ...
                        && ( norm(translation_vec)>max(30,min(last_pred([3 4])+5)) || norm(translation_vec)>max(30,0.7*min(last_pred([3 4]))) && my.occ(seq.frame-1)) ...
                        || ( seq.frame>20 && mean(my.score(seq.frame-20:seq.frame-1,2))>0.08 && norm(translation_vec)>max(10,0.7*min(last_pred([3 4]))) && my.score(seq.frame,2)<0.05 )
                    pos = last_pos;
                    flag_update = false;
                end
            case '010'
                my.chose(seq.frame) = 3;
                tmp = [gmc_pos(1)+(last_pred(4)-1)/2, gmc_pos(2)+(last_pred(3)-1)/2];
                if my.score(seq.frame,2)<0.08 && (my.motion(seq.frame,1)>3*motion_new_pos || my.motion(seq.frame,1)>30 && my.score(seq.frame,2)<0.05) ...
                        && (flag_large || hit_obj || ...
                        my.score(seq.frame,1)>my.score(seq.frame,2) ...
                        && ( max(response_gmc(:))>0.04 && norm(tmp-pos)<50 && max(response_gmc(:))>=my.score(seq.frame,2)+0.005 || my.score(seq.frame,1)>0.04 && sum(my.score(1:seq.frame,4))>1 || my.score(seq.frame,5)>my.score(seq.frame,2) || my.score(seq.frame,1)>=0.08 ) ) ...
                        || (my.score(seq.frame,2)<0.05 && seq.frame>40 && std(my.gmc_box(1:seq.frame,1))<30 && std(my.gmc_box(1:seq.frame,2))<30)
                    if ~(seq.frame<100 && my.score(seq.frame,1)<0.05)  && ~(seq.frame<15 && norm(tmp-pos)>100) && ~( my.score(seq.frame,2)>0.04 && norm(tmp-size(im,[1 2])/2)>1.5*norm(pos-size(im,[1 2])/2) && norm(tmp-size(im,[1 2])/2)>0.3*min(size(im,[1 2])) ) ...
                            && ~(norm(pos-size(im,[1 2])/2)<norm(tmp-size(im,[1 2])/2) && (tmp(1)<0.15*size(im,1) || tmp(1)>0.85*size(im,1) || tmp(2)<0.15*size(im,2) || tmp(2)>0.85*size(im,2)))
                        force_update = true;
                        pos = tmp;
                        currentScaleFactor = currentScaleFactor_ori;
                    end
                end
            case '011'
                tmp1 = [gmc_pos(1)+(last_pred(4)-1)/2, gmc_pos(2)+(last_pred(3)-1)/2];
                tmp2 = last_pos+[dy_lf,dx_lf];
                if calcRectInt(my.traj_box(seq.frame,:),my.gmc_box(seq.frame,:))>0.5
                    pos = tmp1;
                    flag_merge = true;
                elseif norm(tmp1-tmp2)>max(last_pred([3 4]))
                    pos = tmp2;
                elseif calcRectInt(my.traj_box(seq.frame,:),my.gmc_box(seq.frame,:))>0
                    pos = (tmp1+tmp2)/2;
                end
                currentScaleFactor = currentScaleFactor_ori;
                if my.score(seq.frame,3)<my.score(seq.frame,2) && norm(translation_vec)>20
                    strong_update = true;
                end
            case '001'
                if prod(last_pred([3 4]))>2000 && is_color_image && mean(my.score(1:20,2))>0.1 && ~flag_large && (seq.frame>10 && mean(my.score(seq.frame-10:seq.frame,2))>0.06 && my.score(seq.frame,2)-my.score(seq.frame-1,2)<0.01 && (my.score(seq.frame,2)<0.05) ||...
                        (prod(last_pred([3 4]))>4500 && seq.frame>20 && mean(my.score(seq.frame-20:seq.frame,2))>0.08 && my.score(seq.frame,2)<0.08) )
                    tmp = myMeanColor(im,pos,base_target_sz * currentScaleFactor);
                    tmp = squeeze(tmp)';
                    tmp_dist = mean(abs(tmp-my.meanColor_trust(seq.frame-10,:)),'all');
                    if tmp_dist>20 || (tmp_dist>10 && my.score(seq.frame,2)-my.score(seq.frame-5,2)<-0.1)
                        flag_occ = true;
                        my.occ(seq.frame) = 1;
                        flag_update = false;
                        currentScaleFactor = currentScaleFactor_ori;
                    end                    
                end
                color1 = squeeze(myMeanColor(im,pos,base_target_sz * currentScaleFactor))';
                dist1 = mean(abs(color1-my.meanColor_trust(1,:)),'all');
                tmp = last_pos+[dy_lf,dx_lf];
                color2 = squeeze(myMeanColor(im,tmp,base_target_sz * currentScaleFactor))';
                dist2 = mean(abs(color2-my.meanColor_trust(1,:)),'all');
                if norm(translation_vec)>20 && ~((dist1<dist2-8 || dist1<15) && my.score(seq.frame,2)>my.score(seq.frame,3))
                    my.chose(seq.frame) = 2;
                    pos = tmp;
                    currentScaleFactor = currentScaleFactor_ori;
                    if ~flag_occ
                        strong_update = true;
                    end
                end
            case '101'
                if my.score(seq.frame,3)>my.score(seq.frame,2) && norm(translation_vec)>10
                    tmp = last_pos+[dy_lf,dx_lf];
                    pos = tmp;
                    my.chose(seq.frame) = 1.5;
                else
                    flag_merge = true;
                end
        end
    end
    
    % box
    [m,n] = size(im,[1 2]);
    tmp_sz = base_target_sz * currentScaleFactor;
    my.pos_sz(seq.frame,:) = [pos(2) pos(1) tmp_sz(2) tmp_sz(1)];
    pred = [pos(2)-tmp_sz(2)/2, pos(1)-tmp_sz(1)/2, pos(2)+tmp_sz(2)/2, pos(1)+tmp_sz(1)/2];
    % no update if box out of frame
    if seq.frame>1 && (min(pred)<1 || max([pred(1),pred(3)])>n || max([pred(2),pred(4)])>m)
        flag_update = false;
    end
    
    % color
    if is_color_image==true
        if seq.frame==1
            init_vec = get_color_hist(im,pos,tmp_sz);
            my.color_hist_dist = ones(seq.len,3);
            my.color_hist_dist(1,:) = 0;
        else
            vec = get_color_hist(im,pos,tmp_sz);
            my.color_hist_dist(seq.frame,1) = pdist2(init_vec,vec,'chisq');
            if ~isempty(gmc_pos)
                vec = get_color_hist(im,gmc_pos,tmp_sz);
                my.color_hist_dist(seq.frame,2) = pdist2(init_vec,vec,'chisq');
            end
        end
    end
    
    % loss detection
    flag_loss = false;
    if is_color_image==true
        flag1 = seq.frame>100 && std(my.pos_sz(seq.frame-100:seq.frame-1,1))<3 && std(my.pos_sz(seq.frame-100:seq.frame-1,2))<3 ...
                && ~(std(my.pos_sz(1:seq.frame-1,1))<10 && std(my.pos_sz(1:seq.frame-1,2))<10 && prod(last_pred([3 4]))<150*150);
        flag2 = my.color_hist_dist(seq.frame,1)>my.color_hist_dist(seq.frame,2)+0.2 && my.color_hist_dist(seq.frame,2)<0.2;
        if ~flag_occ && ~force_update && flag_detect && ~isempty(gmc_pos) && prod(last_pred([3 4]))>225 && ...
                (flag1 || flag2) 
            tmp = gather(sum(bsxfun(@times, conj(cf_f_init{k1}), xtf_gmc{k1}), 3));
            tmp = gather(tmp);
            tmp = resizeDFT2(tmp, output_sz);
            tmp = ifft2(tmp, 'symmetric');
            score_gmc = tmp(1,1);
            tmp = gather(sum(bsxfun(@times, conj(cf_f_init{k1}), xtf{k1}(:,:,:,3)), 3));
            tmp = gather(tmp);
            tmp = resizeDFT2(tmp, output_sz);
            tmp = ifft2(tmp, 'symmetric');
            score_pos = tmp(1,1);
            if (score_gmc>0.03 && ~flag1 || flag1) && score_gmc>score_pos && score_pos<0.04 && ~(my.color_hist_dist(seq.frame,2)>my.color_hist_dist(seq.frame,1)+0.2 && my.color_hist_dist(seq.frame,1)<0.1) ...
                    || score_gmc>0.03 && score_gmc+0.0005>score_pos && (my.color_hist_dist(seq.frame,1)>my.color_hist_dist(seq.frame,2)+0.2 && my.color_hist_dist(seq.frame,2)<0.1)
                pos = [gmc_pos(1)+(last_pred(4)-1)/2, gmc_pos(2)+(last_pred(3)-1)/2];
                flag_loss = true;
                flag_update = true;
                %cf_f = cf_f_init;
            end
        end
    end
    
    
    flag_use_clf = false;
    cf_sim_score = my.score(seq.frame,2);
    pred_seg = [];
    try
        if (flag_seg || flag_seg_small) && cf_sim_score<0.2 && ~flag_loss && ~flag_occ && ~force_update && ~strong_update
            current_pred = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
            res = py.segment.process_frame(seq.image_files{seq.frame},py.list(current_pred),init_colorkey_info,seg_mode);
            if logical(res{1})
                patch = double(res{2});
                fg_coors = cell(res{4});
                fg_coors = {double(py.array.array('d',fg_coors{1}))+1, double(py.array.array('d',fg_coors{2}))+1};
                bg_coors = cell(res{5});
                bg_coors = {double(py.array.array('d',bg_coors{1}))+1, double(py.array.array('d',bg_coors{2}))+1};
                [bayes_seg,final_seg] = mrf(patch,fg_coors,bg_coors,2,10,10,3,false);
                
                res = py.segment.get_box(final_seg,res{3},py.list(current_pred));
                pred_seg_scaled = double(py.array.array('d',res{1}));
                pred_seg = double(py.array.array('d',res{2}));
                if params.visualization
                    figure(2);
                    subplot(1,2,1),imshow(patch/255);
                    subplot(1,2,2),hold on;
                    imshow(final_seg);
                    rectangle('Position',pred_seg_scaled, 'EdgeColor','g', 'LineWidth',2);
                    hold off;
                    tmp = calcRectInt(pred_seg,current_pred);
                    display(['frame ' num2str(seq.frame) ' iou:' num2str(tmp)]);
                end
            end
        end
    catch ME
        [];
    end
    
    if ~isempty(pred_seg)
        my.seg_box(seq.frame,:) = pred_seg;
    end
    
    
    
    if seq.frame==1 && (isempty(pred_seg) || ~isempty(pred_seg) && calcRectInt(pred_seg,current_pred)<0.7) || ...
            seq.frame>5 && seq.frame<=10 && ...
            ~any(my.seg_box(seq.frame-5:seq.frame,[3 4])<=0,'all') && (std(my.seg_box(seq.frame-5:seq.frame,3))>0.2*target_sz(2) || std(my.seg_box(seq.frame-5:seq.frame,4))>0.2*target_sz(1))
        flag_seg = false;
        flag_seg_small = false;
    end
    
    flag_refine_seg = false;
    if (flag_seg || flag_seg_small) && ~isempty(pred_seg)&& pred_seg(3)>0 && pred_seg(4)>0 && cf_sim_score<0.2
        vec = get_color_hist(im,pos,tmp_sz);
        color_hist_score1 = pdist2(init_vec,vec,'chisq');
        vec = get_color_hist(im,pred_seg([2 1])+(pred_seg([4 3])-1)/2,pred_seg([4 3]));
        color_hist_score2 = pdist2(init_vec,vec,'chisq');
        my.color_hist_dist(seq.frame,3) = color_hist_score2;
        
        if flag_seg || flag_seg_small
            flag_seg_very_stable = color_hist_score2<0.4 && seq.frame>10 && std(my.seg_box(seq.frame-10:seq.frame,3))<0.05*target_sz(2) && ...
                                                std(my.seg_box(seq.frame-10:seq.frame,4))<0.05*target_sz(1) ;
            if color_hist_score2<color_hist_score1+0.1 && color_hist_score2<0.2 || flag_seg_very_stable
                flag_seg_stable = seq.frame>5 && std(my.seg_box(seq.frame-5:seq.frame,3))<0.05*target_sz(2) && ...
                                                std(my.seg_box(seq.frame-5:seq.frame,4))<0.05*target_sz(1) ;
                
                if (flag_seg || flag_seg_small && flag_seg_stable) && calcRectInt(pred_seg,current_pred)>0.85 || ...
                        (flag_seg_stable && calcRectInt(pred_seg,current_pred)<0.6 && calcRectInt(pred_seg,current_pred)>0.3)
                    flag_use_clf = true;
                    if calcRectInt(pred_seg,current_pred)<0.6
                        strong_update = true;
                    end
                    target_sz = pred_seg([4 3]);
                    pos = pred_seg([2 1]) + (target_sz - 1)/2;% [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])]
                    currentScaleFactor = sqrt(prod(target_sz*params.search_area_scale)/prod(img_sample_sz));

                    if nScales > 0
                        %force reasonable scale changes
                        min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
                        max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
                    end
                    % Adjust to make sure we are not to large or to small
                    if currentScaleFactor < min_scale_factor
                        currentScaleFactor = min_scale_factor;
                    elseif currentScaleFactor > max_scale_factor
                        currentScaleFactor = max_scale_factor;
                    end

                    base_target_sz = target_sz/currentScaleFactor;
                end
            end

            if flag_seg && ~flag_use_clf && color_hist_score2<color_hist_score1+0.01 && color_hist_score2<0.4
                pos = pred_seg([2 1]) + (target_sz - 1)/2;% [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])]
                flag_refine_seg = true;
            end
        end
    end
    
    my.meanColor(seq.frame,:) = myMeanColor(im,pos,base_target_sz * currentScaleFactor);
    if flag_occ
        my.meanColor_trust(seq.frame,:) = my.meanColor_trust(seq.frame-3,:);
    else
        my.meanColor_trust(seq.frame,:) = my.meanColor(seq.frame,:);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if flag_update
        % extract image region for training sample
        sample_pos = round(pos);
        [xl,patches] = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);

        % do windowing of features
        xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);

        % compute the fourier series
        xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
        
        % Update the target size (only used for computing output box)
        target_sz = base_target_sz * currentScaleFactor;

        % train the CF model for each feature
        for k = 1: numel(xlf)
            model_xf = xlf{k};

            if (seq.frame == 1)
                f_pre_f{k} = zeros(size(model_xf));
                mu = 0;
            elseif flag_loss
                mu = 1;
            elseif strong_update
                if my.score(seq.frame,2)>0.05
                    mu = 5;
                else
                    mu = 1;
                end 
            elseif force_update || ~isempty(gmc_pos) && norm(pos-(target_sz-1)/2-gmc_pos)<max([5,0.1*min(target_sz)]) && my.motion(seq.frame,1)>10 && my.score(seq.frame,2)>0.1
                if my.score(seq.frame,2)>0.05
                    mu = 10;
                else
                    mu = 5;
                end
            else
                mu = temporal_regularization_factor(k);
            end
            my.update_rate(seq.frame) = mu;

            % intialize the variables
            f_f = single(zeros(size(model_xf)));
            g_f = f_f;
            h_f = f_f;
            gamma  = init_penalty_factor(k);
            gamma_max = max_penalty_factor(k);
            gamma_scale_step = penalty_scale_step(k);

            % use the GPU mode
            if params.use_gpu
                model_xf = gpuArray(model_xf);
                f_f = gpuArray(f_f);
                f_pre_f{k} = gpuArray(f_pre_f{k});
                g_f = gpuArray(g_f);
                h_f = gpuArray(h_f);
                reg_window{k} = gpuArray(reg_window{k});
                yf{k} = gpuArray(yf{k});
            end

            % pre-compute the variables
            T = prod(output_sz);
            S_xx = sum(conj(model_xf) .* model_xf, 3);
            Sf_pre_f = sum(conj(model_xf) .* f_pre_f{k}, 3);
            Sfx_pre_f = bsxfun(@times, model_xf, Sf_pre_f);

            % solve via ADMM algorithm
            iter = 1;
            while (iter <= admm_max_iterations)

                % subproblem f
                B = S_xx + T * (gamma + mu);
                Sgx_f = sum(conj(model_xf) .* g_f, 3);
                Shx_f = sum(conj(model_xf) .* h_f, 3);

                f_f = ((1/(T*(gamma + mu)) * bsxfun(@times,  yf{k}, model_xf)) - ((1/(gamma + mu)) * h_f) +(gamma/(gamma + mu)) * g_f) + (mu/(gamma + mu)) * f_pre_f{k} - ...
                    bsxfun(@rdivide,(1/(T*(gamma + mu)) * bsxfun(@times, model_xf, (S_xx .*  yf{k})) + (mu/(gamma + mu)) * Sfx_pre_f - ...
                    (1/(gamma + mu))* (bsxfun(@times, model_xf, Shx_f)) +(gamma/(gamma + mu))* (bsxfun(@times, model_xf, Sgx_f))), B);

                %   subproblem g
                g_f = fft2(argmin_g(reg_window{k}, gamma, real(ifft2(gamma * f_f+ h_f)), g_f));

                %   update h
                h_f = h_f + (gamma * (f_f - g_f));

                %   update gamma
                gamma = min(gamma_scale_step * gamma, gamma_max);

                iter = iter+1;
            end

            % save the trained filters
            f_pre_f{k} = f_f;
            cf_f{k} = f_f;
        end  
    end
    
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    my.output_box(seq.frame,:) = seq.rect_position(seq.frame,:);
    flag_no_refine = true;
    if ~flag_no_refine && seq.frame>10 && flag_seg_sp && ~flag_occ && gmc_success && cf_sim_score<0.2
       % try superpixel seg
       current_pred = seq.rect_position(seq.frame,:);
        pred_bb = current_pred;
        try
            reg_res = py.heatmap_detection.regress_box_gmc(dif_map_py,py.numpy.array(current_pred/dif_ds_stride),2.5,0.2);
            gmc_seg_box = double(reg_res{1});
            gmc_seg_motionscore = getScoreInBox_m(dif_map,gmc_seg_box);
        catch ME
            gmc_seg_box = [0,0,0,0];
            gmc_seg_motionscore = 0;
        end
        gmc_seg_box_upscale = dif_ds_stride*gmc_seg_box + [1 1 0 0];
        gmc_seg_box = gmc_seg_box + [1 1 0 0];

        if calcRectInt(gmc_seg_box_upscale,current_pred)>0 && gmc_seg_motionscore>4
            if prod(gmc_seg_box_upscale([3 4]))>0 && ~(gmc_seg_box_upscale(4)>4*gmc_seg_box_upscale(3) || gmc_seg_box_upscale(4)<0.25*gmc_seg_box_upscale(3))
                if prod(gmc_seg_box_upscale([3 4]))>0 && calcRectInt(gmc_seg_box_upscale,current_pred)>0
                    sample_box = myMinRec(gmc_seg_box_upscale,current_pred);
                else
                    sample_w = max([current_pred(3),gmc_seg_box_upscale(3)]);
                    sample_h = max([current_pred(4),gmc_seg_box_upscale(4)]);
                    tmp = convert_bbox_format_m(current_pred,'center');
                    cx = tmp(1);
                    cy = tmp(2);
                    sample_box = convert_bbox_format_m([cx,cy,sample_w,sample_h],'topleft');
                end
            else
                sample_box = current_pred;
            end

            % get patch - slow
            res = py.segment.getPatch_matlab(seq.image_files{seq.frame},py.list(sample_box),int32(32),int32(48));
            patch_py = res{1};
            patch = uint8(double(res{1}));
            patch_scale = res{2};

            % scale the boxes
            gmc_seg_box_scaled = getScaledBox_m(sample_box,gmc_seg_box_upscale,patch_scale,[size(patch,1),size(patch,2)]);
            pred_bb_scaled = getScaledBox_m(sample_box,current_pred,patch_scale,[size(patch,1),size(patch,2)]);
            sample_box_scaled = getScaledBox_m(sample_box,sample_box,patch_scale,[size(patch,1),size(patch,2)]);

            if ~any(pred_bb_scaled<=0)
                % seg
                d_sigma              = 0.6;   % default: 0.5
                i_k                  = 500;   % default: 500
                i_minSize            = 50;    % default: 50
                b_computeColorOutput = false;  % default: false
                s_destination        = '';    % default: ''
                b_verbose            = false; % default: false

                segments = segmentFelzenszwalb( patch, d_sigma, i_k, i_minSize, ...
                                                 b_computeColorOutput, ...
                                                 s_destination, ...
                                                 b_verbose...
                                               );

                % generate box
                boxes = py.list();
                boxes.append(pred_bb_scaled);
                boxes.append(gmc_seg_box_scaled);

                res = py.segment.box_refine_segments(patch_py,segments,sample_box_scaled,boxes);
                sp_seg_box_scaled = double(res{1});
                tmp_std = res{2};

                sp_seg_box = getFrameBox_m(sp_seg_box_scaled,patch_scale,sample_box,[size(patch,1),size(patch,2)]);
                iou1 = calcRectInt(current_pred,sp_seg_box);
                iou2 = calcRectInt(gmc_seg_box_upscale,sp_seg_box);
                if calcRectInt(gmc_seg_box_upscale,current_pred)>0.7 || iou1>0.7
                    numGoodIouMotion = numGoodIouMotion + 1;
                end
                if numGoodIouMotion>=5
                    if tmp_std>0
                        if min([iou1 iou2])>0.3 || iou1>0.7 || iou2>0.4 && gmc_seg_motionscore>10
                            pred_bb = sp_seg_box;
                        end
                    elseif iou1>0.7 || iou2>0.7 && gmc_seg_motionscore>10
                        pred_bb = sp_seg_box;
                    end
                end
                
            end
        end 
        my.output_box(seq.frame,:) = pred_bb;
    end
    
    if seq.frame>1 && (flag_seg || flag_seg_small)
        if flag_refine_seg
            my.output_box(seq.frame,:) = (pred_seg+my.output_box(seq.frame-1,:))/2;
        elseif ~flag_use_clf && flag_seg && ~isempty(pred_seg) && cf_sim_score<0.2 && color_hist_score2<color_hist_score1+0.01 &&...
                (prod(pred_seg([3 4]))<100*100 && calcRectInt(pred_seg,current_pred)>0.85 || ...
                prod(pred_seg([3 4]))>=100*100 && prod(pred_seg([3 4]))<200*200 && calcRectInt(pred_seg,current_pred)>0.7 || ...
                prod(pred_seg([3 4]))>=200*200 && calcRectInt(pred_seg,current_pred)>0.6)
            my.output_box(seq.frame,:) = pred_seg;
        elseif ~flag_use_clf && flag_seg_small && ~isempty(pred_seg) && cf_sim_score<0.2 && color_hist_score2<color_hist_score1+0.01 &&...
                calcRectInt(pred_seg,current_pred)>0.9
            my.output_box(seq.frame,:) = pred_seg;
        end
    end
    my.output_box_ori(seq.frame,:) = seq.rect_position(seq.frame,:);
    
    %my
    if seq.frame==1
        cf_f_init = cf_f;
        last_op_model = cf_f;
        my.score(seq.frame,4) = 0.5;
    elseif ~isempty(gmc_pos) && norm(pos-(target_sz-1)/2-gmc_pos)<max([5,0.1*min(target_sz)])
        last_op_model = cf_f;
        my.score(seq.frame,4) = 0.5;
    end
    if seq.frame>1
        last_im_gray = curr_im_gray;
    end
    tmp = seq.rect_position(seq.frame,:);
    if gmc_success
        [my.motion(seq.frame,3),~] = getValue_my(dif_int,int32(tmp(2)/dif_ds_stride), int32(tmp(1)/dif_ds_stride));
    end
    if seq.frame==6
        tmp = true;
        for i=2:6
            if calcRectInt(seq.rect_position(i,:),my.gmc_box(i,:))<0.5
                tmp = false;
                break;
            end
        end
        if tmp
            hit_obj = true;
            disp('main moving obj');
        end
    end
    if hit_obj && seq.frame>10
        if std(my.gmc_box(seq.frame-10:seq.frame,1))>60 || std(my.gmc_box(seq.frame-10:seq.frame,2))>60
            hit_obj = false;
            disp('not main moving obj now');
        end
    end
    if seq.frame==20 && flag_large && sum(my.rapid_change(1:20))>=10
        flag_large = false;
    end
    
    if flag_lv && my.no_gmc_cnt>-1
        my.no_gmc_cnt = my.no_gmc_cnt+1;
        if my.no_gmc_cnt==10
            flag_lv = false;
            my.no_gmc_cnt = -1;
        end
    end
        
    
    seq.time = seq.time + toc();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % visualization
    if params.visualization
        rect_position_vis = [pos_ori([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        
        figure(1);
        imagesc(im_to_show);
        hold on;

        rectangle('Position',rect_position_vis, 'EdgeColor','y', 'LineWidth',2);
        
        if seq.frame>1 && ~isempty(gmc_pos)
            rectangle('Position',[gmc_pos(2), gmc_pos(1), w, h], 'EdgeColor','m', 'LineWidth',2);
        end
        if ~isempty(dx_lf)
            tmp = seq.rect_position(seq.frame-1,:);
            rectangle('Position',[tmp(1)+dx_lf, tmp(2)+dy_lf, tmp(3), tmp(4)], 'EdgeColor','b', 'LineWidth',2);
        end
        
        rectangle('Position',my.output_box(seq.frame,:), 'EdgeColor','g', 'LineWidth',2);
        
        text(10, 10, [int2str(seq.frame) '/'  int2str(size(seq.image_files, 1))], 'color', [0 1 1]);
        hold off;
        axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
                    
        drawnow
    end
end

seq.rect_position = my.output_box;
[~, results] = get_sequence_results(seq);
