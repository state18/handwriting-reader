function emnist2png(csvPath, outDir)

% Read csv file.
c = csvread(csvPath);

% Create output directory for images.
mkdir(outDir);


rowBeginVals = 1:28:784;
rowEndVals = 28:28:784;
% Each row is data for a 28x28 image
% First column is label information. The rest are image data.

for i = 1:size(c, 1)
    imgLabel = c(i, 1);
    imgData = c(i, 2:end);
    
    img = uint8(zeros(28, 28, 3));
    for j = 1:28
        dataSlice = imgData(rowBeginVals(j):rowEndVals(j));
        for k = 1:3
            img(:, j, k) = dataSlice;
        end
    end
    
    saveDir = sprintf('%s/%d', outDir, imgLabel);
    mkdir(saveDir);
    imwrite(img, sprintf('%s/%d_%05d.png', saveDir, imgLabel, i));

end

