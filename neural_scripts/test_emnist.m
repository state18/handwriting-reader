%% Note: This is the working file.

%% 0.  my parameters
trainFrac = 0.99; % rest goes to valid.
sampleFrac = 1; % 0.01 means use 1%
testFrac = .01;

% dataPath = '~/patch_data/baseline_2tracklets/baseline281/';
% dataPath = 'results/Channelwise32/baseline281_2-segments/';
dataPath = 'C:\Users\samta\school\6112\data\emnist\png\';
dataPath = 'C:\Users\samta\school\6112\data\chars74k\EnglishFnt\EnglishFnt\English\';
savePath = 'C:\Users\samta\school\6112\neural_scripts\nets\';
labelKeyTrainPath = 'C:\Users\samta\school\6112\neural_scripts\mat\emnist_merge_train_labels.mat';
labelKeyTestPath = 'C:\Users\samta\school\6112\neural_scripts\mat\emnist_merge_test_labels.mat'; % Same as train for now!
trainFolder = [dataPath, 'merged-train-rgb'];
trainFolder = dataPath;
testFolder = [dataPath, 'merged-train-rgb']; % Same for now... just on local machine
testFolder = '';

whichNetwork = 'tutorialNet'; %'pb2';
inputSize = [28, 28, 3];
inputSize = [128, 128];

% If using cluster computing, use the values passed in.
if exist('cTrainFrac', 'var')
    trainFrac = cTrainFrac;
    sampleFrac = cSampleFrac;
    
    trainFolder = regexprep(char(cTrainPath), '\', '/');
    testFolder = regexprep(char(cTestPath), '\', '/');
    
    whichNetwork = cWhichNetwork;
    
    disp(trainFolder)
    disp(testFolder)
    
end

if exist('cInputSize', 'var')
	inputSize = cInputSize;
end

% Determine where networks will be saved.
global netPath
netPath = sprintf('~/cnn_pairs/nets_reg/%s/%s/', whichNetwork, regexprep([trainFolder, testFolder], '[/~]', '_'));
%local
netPath = 'C:\Users\samta\school\6112\neural_scripts\nets\';
disp(netPath);
mkdir(netPath);


%% 1.  prepare data


% 1) trainData
rootFolder = fullfile ( trainFolder );
trainAllImds = imageDatastore ( rootFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
trainAllImdsCount = numel ( trainAllImds.Files );

if sampleFrac == 1
    trainImds = trainAllImds;   
else
    trainImds = splitEachLabel(trainAllImds, sampleFrac);
end
% TODO: Don't use imbalanced amount of labels.
% tbl = countEachLabel(trainImds);
% minSetCount = min(tbl{:,2});
% minSetCountTrain = minSetCount;

% split for train and valid
[trainImds, valImds] = splitEachLabel ( trainImds, trainFrac );


% trainLabels
trainLabels = trainImds.Labels;


% valLabels
valLabels = valImds.Labels;


%% TEST DATA PREP

if ~strcmp(testFolder, '')
    % 3) test data
    rootFolder = fullfile ( testFolder );
    testImds = imageDatastore ( rootFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    % sample test data
    tbl = countEachLabel(testImds);
    minSetCount = min(tbl{:,2});

    testImds = splitEachLabel(testImds, int32(minSetCount * sampleFrac), 'randomize'); 
    testLabels = testImds.Labels;
end
    


%% 2.  set train options
% Temporary local testing to overfit!
maxEpochs = 20; %14;
miniBatchSize = 128;
numClasses = numel(unique(trainImds.Labels));

options = trainingOptions('sgdm', ...
    'MaxEpochs', maxEpochs,...
    'InitialLearnRate',1e-4, ...
    'MiniBatchSize', miniBatchSize, ...
    'Verbose',1, ...
    'Plots','training-progress');


filterSize = [5 5];
numFilters = 50; 

layers = [ ...
    imageInputLayer(inputSize)
    convolution2dLayer(5,20)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%layers(2).Weights = 0.001 * randn([filterSize 3 numFilters]);
        
                
        
%         outputFunctions = {
%             @stopIfLossNaN,...
%             @(info)stopIfStartingSlowly(info,2,0.1),...
%             @plotTrainingProgress};
        
%             options = trainingOptions('sgdm','InitialLearnRate',0.00000000001, ...
%             'MaxEpochs',15);
            
%%
% Train the network and plot training progress during training.
% Close the plot after training finishes.
[trainedNet,trainInfo] = trainNetwork(trainImds,layers,options);


%%
% Evaluate the trained network on the validation set and calculate
% the validation error.
% predLabels - row = primary key, column = dx or dy
predValLabels = classify(trainedNet, valImds);
valAccuracy = mean(predValLabels == valImds.Labels)

if ~strcmp(testFolder, '')
    predTestLabels = classify(trainedNet, testImds);
    testAccuracy = mean(predTestLabels == testImds.Labels)
else
    testAccuracy = 0;
end

save([savePath, sprintf('net_fnt_all_mini-128_learn-e4__test%d-val%df.mat', ...
    round(testAccuracy * 100), round(valAccuracy * 100))], ...
    'trainedNet', 'trainInfo', 'layers', 'options', 'valAccuracy', 'testAccuracy');
% % %         valAccuracy = mean(predLabels == valTbl{});
%valError = mean(sqrt((predLabels(:,1) - valTbl{:,2}).^ 2 + (predLabels(:,2) - valTbl{:,3}).^ 2));
        

