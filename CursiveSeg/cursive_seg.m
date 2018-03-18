function [output] = cursive_seg(dirName, maxImgs)

if nargin < 2
    maxImgs = Inf;
end

imgsInfo = dir(dirName);
imgsInfo = imgsInfo(3:end);

numImgs = min(numel(imgsInfo), maxImgs);
% Store segmentation columns
segResults = cell(numImgs, 1);

for i_info = 1:numImgs
    imgInfo = imgsInfo(i_info);
    imgPath = [imgInfo.folder, '\', imgInfo.name];
    img = imread(imgPath);
    
    figure;
    subplot(2,2,1), imshow(img);
    
    % Perform preprocessing on image.
    preppedImg = PreprocessImage(img);
    
    
    subplot(2,2,2), imshow(preppedImg);
    
    % Count FG pixels in each column.
    fgCount = sum(preppedImg, 1);
    
    % Note the columns that have 0 or 1 FG pixels
    psc = fgCount <= 1;
    
    pscImg = double(preppedImg);
    pscImg(end, end, 3) = 0;
    pscImg(:, psc, 2) = 255;
    subplot(2,2,3), imshow(pscImg);
    
    % Merge psc columns based on a threshold. (manually tuned for now)
    mergeThresh = 7;
    pscMerged = MergePSCs(fgCount, mergeThresh);
    
    % TODO: Display average merged PSCs.
    pscMergedImg = double(preppedImg);
    pscMergedImg(end, end, 3) = 0;
    pscMergedImg(:, pscMerged, 2) = 255;
    subplot(2,2,4), imshow(pscMergedImg);
    

end

output = 'placeholder';

end

function [outImg] = PreprocessImage(img)

    if size(img,3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end

    %% Binarize
    binImg = imcomplement(imbinarize(grayImg));

    %% Thinning
    %se = strel('line',11,90);
    %shrunk_img = imerode(bin_img, se);
    shrunkImg = bwmorph(binImg, 'skel', inf);

    %% Noise removal
    % TODO
    outImg = shrunkImg;
end

function [pscMerged] = MergePSCs(fgHist, mergeThresh)
% Note, merge thresh currently doubles as max distance to merge pscs and
% min distance a cluster of pscs must span to be considered as 1.

if nargin < 2
    mergeThresh = 7;
end

% Initialize to a large size. Will trim at the end
pscMerged = zeros(100, 1);
numMerged = 0;

currentCluster = zeros(100, 1);
clusterSize = 0;

distFromLast = 0;
% Sweep from left to right.
for i = 1:numel(fgHist)
    currCount = fgHist(i);
    if currCount <= 1
        % Begin new or continue current cluster.
        if distFromLast > mergeThresh
            % Terminate current cluster. Create a new one.
            % If current cluster is smaller than threshold, don't keep it.
            if clusterSize > mergeThresh
                numMerged = numMerged + 1;
                pscMerged(numMerged) = round(mean(currentCluster(1:clusterSize)));
            end
            currentCluster = zeros(100, 1);
            clusterSize = 0;
        end
        % Add to current cluster.
        clusterSize = clusterSize + 1;
        currentCluster(clusterSize) = i;
        distFromLast = 0;
    end
    
    distFromLast = distFromLast + 1;
end

% Wrap up any loose ends.
numMerged = numMerged + 1;
pscMerged(numMerged) = round(mean(currentCluster(1:clusterSize)));

% Trim data structure.
pscMerged(numMerged + 1:end) = [];

end
