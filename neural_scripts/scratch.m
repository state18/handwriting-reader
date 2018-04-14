csvPath = 'C:\Users\samta\school\6112\data\emnist\emnist-letters-train.csv';
c = csvread(csvPath);

for i = 1:20
    figure;
    imshow(reshape(c(i,2:end), [28,28]));
end