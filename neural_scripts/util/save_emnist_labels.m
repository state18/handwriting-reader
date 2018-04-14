function save_emnist_labels(rootDir, savePath)

if nargin < 2
    savePath = 'labelKey.mat';
end

% Labels are subdirectories of depth 1 from rootDir.
dirParts = dir(rootDir);
labelKey = cell(1000000,1);
currNum = 1;
for i = 3:numel(dirParts)

    subName = dirParts(i).name;
    
    subDir = [rootDir, '/', subName];
    numImgs = numel(dir(subDir)) - 2;
    
    labelKey(currNum: currNum + numImgs - 1) = {subName};
    currNum = currNum + numImgs;

end

labelKey(currNum:end) = [];
save(savePath, 'labelKey');

