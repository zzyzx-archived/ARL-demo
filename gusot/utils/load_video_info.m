function [seq, ground_truth] = load_video_info(anno_path,video_path,startframe,endframe)

ground_truth = dlmread([anno_path '.txt']);

seq.format = 'otb';
seq.len = min(size(ground_truth, 1),endframe-startframe+1);
ground_truth = ground_truth(1:seq.len,:);
seq.init_rect = ground_truth(1,:);

img_path = [video_path '/'];

if startframe>1
    idx = startframe:endframe;
else
    idx = 1:seq.len;
end

if exist([img_path num2str(1, '%06i.png')], 'file'),
    img_files = num2str((idx)', [img_path '%06i.png']);
elseif exist([img_path num2str(1, '%06i.jpg')], 'file'),
    img_files = num2str((idx)', [img_path '%06i.jpg']);
elseif exist([img_path num2str(1, '%06i.bmp')], 'file'),
    img_files = num2str((idx)', [img_path '%06i.bmp']);
elseif exist([img_path num2str(1, '%04i.jpg')], 'file'),
    img_files = num2str((idx)', [img_path '%04i.jpg']);
elseif exist([img_path num2str(1, '%08i.jpg')], 'file'),
    img_files = num2str((idx)', [img_path '%08i.jpg']);
else
    error('No image files to load.')
end

seq.s_frames = cellstr(img_files);

end

