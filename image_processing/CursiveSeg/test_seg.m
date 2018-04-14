% This script will read in sample cursive text images and attempt to
% segment the characters (by vertical lines). Algorithm heavily inspired by
% "A New Character Segmentation Approach for Off-Line Cursive Handwritten
% Words"

%% PARAMS

% Directory of cursive word images
sample_dir_name = ...
    'test_imgs';

% Limits number of results
max_imgs = 10;


%% CHARACTER SEGMENTATION
output = cursive_seg(sample_dir_name, max_imgs);


%% CHARACTER RECOGNITION