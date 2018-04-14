function [outInfo] = find_chars(imgPath)
    outInfo.Lines = [];
    outInfo.Words = [];
    outInfo.Characters = [];

    img = imread(imgPath);
    [numRows, numCols] = size(img);

    preppedImg = PreprocessImage(img);


    lines = ExtractLines(preppedImg);
    words = ExtractWords(preppedImg, lines);
    connComp = ComputeConnectedComponentsByWord(preppedImg, lines, words);
    
    connComp = DecomposeConnectedComponents(preppedImg, lines, words, connComp);
    
    % Show lines, words, and characters in a figure.
    VisualizeSegmentation(img, lines, words);
    outInfo.Lines = lines;
    outInfo.Words = words;
    outInfo.Characters = connComp;
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
    % shrunkImg = bwmorph(binImg, 'skel', inf);
    shrunkImg = binImg;
    %% Noise removal
    % TODO
    outImg = shrunkImg;
end


function [lines] = ExtractLines(img)
    % Return n x 3 matrix in form [clusterStart, clusterEnd, clusterMean]

    % Use row histogram to determine line placement.
    rowHist = sum(img, 2);
    % Aggregate clusters of adjadcent FG pixels from histogram.
    lines = FindClusters(rowHist, 2);
end

function [words] = ExtractWords(img, lines)
    % Words will be mapped to lines through a cell array of length lines.
    % Each word row will be 1 x 3 in form [clusterStart, clusterEnd, clusterMean]

    % Extract vertical histogram for each line.
    % Segment based on clusters.
    numLines = size(lines, 1);
    words = cell(numLines, 1);
    for iLine = 1:numLines
        currLine = lines(iLine, :);
        lineStart = currLine(1);
        lineEnd = currLine(2);
        lineImg = img(lineStart:lineEnd, :);
        
        colHist = sum(lineImg, 1);
        words(iLine) = {FindClusters(colHist, 5)};
    end
    
end

function [connComp] = ComputeConnectedComponentsByWord(img, lines, words)
    % Expects a binary image as input along with words structure from
    % 'ExtractWords' function
      
    % For each word bounding box, find connected components.
    numLines = size(lines, 1);
    connComp = cell(numLines, 1);
    for iLine = 1:numLines
        currLine = lines(iLine, :);
        lineRange = currLine(1) : currLine(2);
        currWords = words{iLine};
        numWords = size(currWords, 1);
        
        % Initialize size and struct form by using dummy example.
        clear currWordsComps;
        currWordsComps(numWords) = bwconncomp(img(1,1));
        for iWord = 1:numWords
            currWord = currWords(iWord, :);
            wordRange = currWord(1) : currWord(2);
            wordImg = img(lineRange, wordRange);
            currWordsComps(iWord) = bwconncomp(wordImg);            
        end
        connComp(iLine) = {currWordsComps};
    end
end

function clusters = FindClusters(data, gapThresh)
    
    if nargin < 2
        gapThresh = 0;
    end
    
    % Initialize to a large size. Will trim at the end
    clusters = zeros(100, 3);
    numMerged = 0;

    currentCluster = zeros(100, 1);
    clusterSize = 0;

    distFromLast = 0;
    % Sweep from left to right.
    for i = 1:numel(data)
        currCount = data(i);
        if currCount
            % Begin new or continue current cluster.
            if distFromLast > gapThresh
                % Terminate current cluster. Create a new one.
                % If current cluster is smaller than threshold, don't keep it.
                if clusterSize > gapThresh
                    numMerged = numMerged + 1;
                    clusterStart = currentCluster(1);
                    clusterEnd = currentCluster(clusterSize);
                    clusterMean = round(mean(currentCluster(1:clusterSize)));
                    clusters(numMerged, :) = [clusterStart, clusterEnd, clusterMean];
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
    clusterStart = currentCluster(1);
    clusterEnd = currentCluster(clusterSize);
    clusterMean = round(mean(currentCluster(1:clusterSize)));
    clusters(numMerged, :) = [clusterStart, clusterEnd, clusterMean];

    % Trim data structure.
    clusters(numMerged + 1:end, :) = [];
end

function [connComps] = DecomposeConnectedComponents(img, lines, words, connComps)
    % Outputs n x 1 cell array where n is number of words
    % Cell contents are m x 4 matrix where m is number of characters
    % and 5 represents bbox info and recognition id: [minH, minW, maxH, maxW, id]
    % Note: Recognition id is useful for punctuation info and other special
    % cases and prevents those characters from passing through later classification step.
    
    % TODO
    
    % For each connComp, check if width to height ratio exceeds 80%.
    splitRatio = 1.8;
    
    numLines = size(lines, 1);
    for iLine = 1:numLines
        rowRange = lines(iLine, 1) : lines(iLine, 2);
        rowsHeight = numel(rowRange);
        wordsInfo = words{iLine};
        wordsChars = connComps{iLine};    
        for iWord = 1:numel(wordsChars)
            colRange = wordsInfo(iWord, 1) : wordsInfo(iWord, 2);
            wordImg = img(rowRange, colRange);
            globalOffset = [rowRange(1), colRange(1)];
            comps = wordsChars(iWord);
            
            % Separated components will replace existing components.
            newCompPixels = cell(1, 10000);
            numNewComps = 0;
            
            rProps = regionprops(comps);
            for iComp = 1:numel(rProps)
                localBBox = ceil(rProps(iComp).BoundingBox);
                
                % Don't decompose if width isn't 80% or more of height
                if localBBox(3) < splitRatio * localBBox(4)
                    numNewComps = numNewComps + 1;
                    newCompPixels{numNewComps} = comps.PixelIdxList{iComp};
                    continue;
                end
                               
                % Compute Top-down profile.   
                charPixels = comps.PixelIdxList{iComp};
                
                charImg = wordImg;
                charImg(setdiff(1:numel(charImg), charPixels)) = 0;
                charImg = charImg(rowRange - rowRange(1) + 1, localBBox(1) : localBBox(1) + localBBox(3) - 1);
                
                tdp = zeros(1, size(charImg, 2));
                for pCol = 1:numel(tdp)
                    tdp(pCol) = find(charImg(:, pCol), 1);
                end
                
                try
                    [peakMags, peakLocs] = findpeaks(tdp);
                catch e
                    warning('Not enough data to compute peaks...')
                    peakLocs = [];
                end
                
                if isempty(peakLocs)
                    continue;
                end
                
                % Adjust peak locs to wordImg reference frame.
                peakLocs = peakLocs + localBBox(1) - 1;
                
                % Split along peaks, if and only if resulting patches are
                % not really small (this is a literal edge case)
                              
                prevPeakLoc = localBBox(1);
                peakLocs(end+1) = localBBox(1) + localBBox(3) - 1;
                for iPLoc = 1:numel(peakLocs)
                    currPeak = peakLocs(iPLoc);
                    keepArea = wordImg(:, prevPeakLoc : currPeak);
                    % maxLinIndx = rowsHeight * currPeak;
                    linOffset = rowsHeight * (prevPeakLoc - 1);
                    keepPixels = find(keepArea) + linOffset;
                    
                    numNewComps = numNewComps + 1;
                    newCompPixels{numNewComps} = keepPixels;
                    
                    
                    prevPeakLoc = currPeak;
                end

            end
            % Replace component with new pixel information from
            % decomposition.
            newCompPixels(numNewComps+1:end) = [];
            if numNewComps >= 1
                wordsChars(iWord).PixelIdxList = newCompPixels;
            end
            wordsChars(iWord).NumObjects = size(wordsChars(iWord).PixelIdxList, 2);
        end
        connComps{iLine} = wordsChars;
    end
    
end


function VisualizeSegmentation(img, lines, words, connComp)

    % Draw horizontal lines. (Magenta color)
    img(lines(:, 1), :, 1) = 255;
    img(lines(:, 1), :, 2) = 120;
    img(lines(:, 2), :, 1) = 255;
    img(lines(:, 2), :, 2) = 120;
    
    % Draw vertical lines. (Red color)
    for i = 1:numel(words)
        currLine = lines(i, :);
        rowRange = currLine(1) : currLine(2);
        
        currWords = words{i};
        for j = 1:size(currWords, 1)
            currWord = currWords(j, :);
            
            img(rowRange, currWord(1), 1) = 255;
            img(rowRange, currWord(1), 2) = 0;
            img(rowRange, currWord(1), 3) = 0;
            img(rowRange, currWord(2), 1) = 255;
            img(rowRange, currWord(2), 2) = 0;
            img(rowRange, currWord(2), 3) = 0;
        end
    end
    
    % figure; imshow(img);
    
    % TODO: Show connected components of words. (Likely in different
    % figure region)
    if nargin < 4
        return;
    end
    
    
    
end


