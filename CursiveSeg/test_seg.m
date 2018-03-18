% This script will read in sample cursive text images and attempt to
% segment the characters (by vertical lines). Algorithm heavily inspired by
% "A New Character Segmentation Approach for Off-Line Cursive Handwritten
% Words"

%% PARAMS

% Directory of cursive word images
%sample_dir_name = ...
%    'C:\Users\samta\school\6112\iams\DBs\iamDB\data\words\c01\c01-066\';

sample_dir_name = 'C:\Users\samta\school\6112\paint_data'; 

% Limits number of results
max_imgs = 10;


%% CHARACTER SEGMENTATION
output = cursive_seg(sample_dir_name, max_imgs);


%% CHARACTER RECOGNITION