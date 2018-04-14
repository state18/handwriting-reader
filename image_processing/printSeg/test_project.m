img_path = ...
    'C:\Users\samta\school\6112\image_processing\test_imgs\test_2.png';
% img_path = ...
%     'C:\Users\samta\school\6112\image_processing\test_imgs\handwritten_page_test.png';
    
    
net_path = 'C:\Users\samta\school\6112\neural_scripts\nets\net_all_mini-128_learn-e4__test0-val86f.mat';
% net_path = 'C:\Users\samta\school\6112\neural_scripts\nets\net_fnt_all_mini-128_learn-e4__test0-val85f.mat';

% Get characters from image.
segInfo = find_chars(img_path);

img = imread(img_path);

net = load(net_path);
net = net.trainedNet;
inputSize = net.Layers(1).InputSize(1:2);
% For each character, classify it.
% Note: This code assumes raw connected components as input before planned
% decomposition. It will be heavily reworked. Also, preliminary test
% results show the network always guessing label 21, which is "L". This is
% likely because these images wrap tightly around the character where in
% training it had a lot of space around it.

outString = '';
numLines = size(segInfo.Lines, 1);
for iLine = 1:numLines
    rowRange = segInfo.Lines(iLine, 1) : segInfo.Lines(iLine, 2);
    wordsInfo = segInfo.Words{iLine};
    wordsChars = segInfo.Characters{iLine};    
    for iWord = 1:numel(wordsChars)
        colRange = wordsInfo(iWord, 1) : wordsInfo(iWord, 2);
        wordImg = img(rowRange, colRange, :);
        globalOffset = [rowRange(1), colRange(1)];
        comps = wordsChars(iWord);
        rProps = regionprops(comps);
        for iComp = 1:numel(rProps)
            localBBox = ceil(rProps(iComp).BoundingBox);
            charPixels = comps.PixelIdxList{iComp};
            % charImg = rgb2gray(wordImg);
            charImg = wordImg;
            charImg(setdiff(1:numel(charImg), charPixels)) = 255;
            charImg = charImg(localBBox(2) : localBBox(2) + localBBox(4) - 1, localBBox(1) : localBBox(1) + localBBox(3) - 1);
          
            padAmount = 3;
            paddedPatch = padarray(charImg, [padAmount, padAmount * 2], 255);
            
            %globalBBox = [globalOffset(2) + localBBox(1), globalOffset(1) + localBBox(2), localBBox(3:4)];
            
            % Expand bbox around character.
            %globalBBox = globalBBox + [-5, -5, 10, 10];
            
            %charPatch = imcrop(img, globalBBox);
            %if size(img, 3) == 1
            %    grayPatch = charPatch;
            %else
            %    grayPatch = rgb2gray(charPatch);
            %end
            
            
            
            %binPatch = bwmorph(imcomplement(imbinarize(grayPatch)), 'thicken', 1);
            %padAmount = 3;
            %paddedPatch = padarray(binPatch, [padAmount, padAmount * 2], 0);
            %resizedPatch = imresize(paddedPatch, inputSize);
            resizedPatch = imresize(paddedPatch, inputSize);
            % Back to rgb for now......
            rgbPatch = uint8(zeros(size(resizedPatch,1), size(resizedPatch,2),3));
            for iChannel = 1:3
               rgbPatch(:,:,iChannel) = resizedPatch;
            end
            % Classify patch.
            %c = classify(net, uint8(rgbPatch));
            %figure; imshow(rgbPatch); title(string(c));
            
             c = classify(net, imcomplement(rgbPatch));
            %c = classify(net, rgb2gray(rgbPatch));
            %figure; imshow(imcomplement(rgbPatch)); title(emnist_lookup(str2double(string(c))));
            outString(end+1) = emnist_lookup(str2double(string(c)));
        end
        outString(end+1) = ' ';
    end
end

disp(outString);
