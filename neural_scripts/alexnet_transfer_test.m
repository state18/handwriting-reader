%% Load Data

dataPath = 'C:\Users\samta\school\6112\data\emnist\png\merged-train-rgb';
savePath = 'C:\Users\samta\school\6112\neural_scripts\nets';
sampleFrac = .01;

imds = imageDatastore(dataPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Reduce by sampleFrac
if sampleFrac ~= 1
    imds = splitEachLabel(imds, sampleFrac);
end


% % unzip('MerchData.zip');
% % imds = imageDatastore('MerchData', ...
% %     'IncludeSubfolders',true, ...
% %     'LabelSource','foldernames');

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
numTrainImages = numel(imdsTrain.Labels);


%% Initialize/Train network
net = alexnet;

inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

pixelRange = [-1 1];
imageAugmenter = imageDataAugmenter( ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);

%% Classify validation images
[YPred,scores] = classify(netTransfer,augimdsValidation);

% Visualize some predictions.
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

%% Gather accuracy metrics.
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)




