clear;
close all;

pic_name = 'person';
im = imread(['/home/zhiruo/Documents/VOT/code/Ours/utils/MRF_BCD-master/' pic_name '.png']);
fg = imread(['/home/zhiruo/Documents/VOT/code/Ours/utils/MRF_BCD-master/' pic_name '_fg.png']);
bg = imread(['/home/zhiruo/Documents/VOT/code/Ours/utils/MRF_BCD-master/' pic_name '_bg.png']);
resize_target = [64 64];
im = mexResize(im,resize_target,'auto');
fg = mexResize(fg,resize_target,'auto');
bg = mexResize(bg,resize_target,'auto');
[fg_r,fg_c] = ind2sub(size(fg),find(fg>0));
fg_coors = {fg_r,fg_c};
[bg_r,bg_c] = ind2sub(size(bg),find(bg>0));
bg_coors = {bg_r,bg_c};

n_compo = 2;
max_iter = 10;
lmbd = 10;
lmbd2 = 5;
verbose = true;

tic;
[bayes_seg,final_seg] = mrf(double(im),fg_coors,bg_coors,n_compo,max_iter,lmbd,lmbd2,verbose);
toc;
figure;
subplot(1,2,1),imshow(bayes_seg);
subplot(1,2,2),imshow(final_seg);