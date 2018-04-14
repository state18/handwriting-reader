function [outImages] = six_channel_reader(imgPath)

grayImage = rgb2gray(imread(imgPath));
if size(grayImage, 3) == 3
    grayImage = rgb2gray(grayImage);
end

outImages = zeros([size(grayImage), 6]);
for i = 1:6
    outImages(:, :, i) = grayImage;
end

end

