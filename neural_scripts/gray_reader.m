function [rgbImage] = gray_reader(imgPath)
grayImage = imread(imgPath);
rgbImage = cat(3, grayImage, grayImage, grayImage);
end

