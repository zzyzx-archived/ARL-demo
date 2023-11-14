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
flag_occ_cnt = 0;

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

% Check if color image
% if size(im,3) == 3
%     if all(all(im(:,:,1) == im(:,:,2)))
%         is_color_image = false;
%     else
%         is_color_image = true;
%     end
% else
%     is_color_image = false;
% end

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
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% seg
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if seq.frame==1
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
%         if ~flag_seg
%             %ground_truth = dlmread([anno_path '.txt']);
%             results.res = [];
%             return;
%         end
        if target_sz(1)<=50 && target_sz(2)<=50
            flag_seg = false;
            flag_seg_small = true;
        %else
        %    results.res = [];
        %    return;
        end
        init_colorkey_info = res{2};
        fprintf('init box size (%d,%d), %d\n', target_sz(1), target_sz(2), prod(target_sz)>200*200);
    end
    
    current_pred = seq.prev_result(seq.frame,:);
    if seq.frame>10 && (flag_seg || flag_seg_small)
        results.res = [];
        return;
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
        last_pred = seq.prev_result(seq.frame-1,:);
        flag_small = prod(last_pred([3 4]))<1000;
    end
    if seq.frame>1 && ~flag_lv
        w = int32(last_pred(3));
        h = int32(last_pred(4));
        
        tStart = cputime;
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
        
        my.timer(seq.frame,1) = cputime - tStart;
        res=cell(res);
        gmc_success=logical(res{1});
        dif_map = double(res{4});
        dif_map_py = res{4};
    end
    
    flag_occ = false;
    
    % loss detection
    flag_loss = false;
    
    flag_use_clf = false;
    cf_sim_score = my.score(seq.frame,2);
    pred_seg = [];
    try
        if (flag_seg || flag_seg_small) && cf_sim_score<0.2 && ~flag_loss && ~flag_occ && ~force_update && ~strong_update
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
        %results.res = [];
        %return;
    end
    
    
    
    % try superpixel seg
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
            if tmp_std>0
                if prod(sp_seg_box([3 4]))<prod(current_pred([3 4])) && iou1>0.3 || prod(sp_seg_box([3 4]))<prod(gmc_seg_box_upscale([3 4])) && iou2>0.3 ...
                    || prod(sp_seg_box([3 4]))>prod(current_pred([3 4])) && iou1>0.7 || prod(sp_seg_box([3 4]))>prod(gmc_seg_box_upscale([3 4])) && iou2>0.4 && gmc_seg_motionscore>10 || tmp_std<25 && (sp_seg_box(3)<size(patch,2) && sp_seg_box(4)<size(patch,1))
                    pred_bb = sp_seg_box;
                end
            elseif iou1>0.7 || iou2>0.7 && gmc_seg_motionscore>10
                pred_bb = sp_seg_box;
            end
        end
    end
    
    % refine output box
    my.output_box(seq.frame,:) = pred_bb;
    
    %my
    if seq.frame>1
        last_im_gray = curr_im_gray;
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
        rect_position_vis = current_pred;
        im_to_show = double(im)/255;
        
        figure(1);
        imagesc(im_to_show);
        hold on;
%         if isfield(seq,'gt_boxes')
%             try
%                 rectangle('Position',seq.gt_boxes(seq.frame,:), 'EdgeColor','g', 'LineWidth',2);
%             catch err
%                 %disp('err showing gt box');
%             end
%         end
        rectangle('Position',rect_position_vis, 'EdgeColor','y', 'LineWidth',2);
%         rectangle('Position',my.output_box_ori(seq.frame,:), 'EdgeColor','b', 'LineWidth',2);
        
        
%         if seq.frame>1 && ~isempty(gmc_seg_box)
%             rectangle('Position',gmc_seg_box_upscale, 'EdgeColor','m', 'LineWidth',2);
%         end
        
        rectangle('Position',my.output_box(seq.frame,:), 'EdgeColor','g', 'LineWidth',2);
        
        text(10, 10, [int2str(seq.frame) '/'  int2str(size(seq.image_files, 1))], 'color', [0 1 1]);
        hold off;
        axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
                    
        drawnow
        
%         if seq.frame>0
%             saveas(gcf,['./video/',seq.name,'/',sprintf('%04d.jpg',seq.frame)]);
%         end
    end
end

seq.rect_position = my.output_box;
[~, results] = get_sequence_results(seq);

disp(['fps: ' num2str(results.fps)])
