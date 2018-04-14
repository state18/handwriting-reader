% This script is for doing regression on two variables, <dx,dy>

%% 0.  my parameters
trainFrac = 0.7; % rest goes to valid.
sampleFrac = 1; % 0.01 means use 1%
testFrac = .01;

% dataPath = '~/patch_data/baseline_2tracklets/baseline281/';
% dataPath = 'results/Channelwise32/baseline281_2-segments/';
dataPath = 'C:\Users\samta\school\6112\data\emnist\png\merged\';
labelKeyTrainPath = 'C:\Users\samta\school\6112\neural_scripts\mat\emnist_merge_train_labels.mat';
labelKeyTestPath = 'C:\Users\samta\school\6112\neural_scripts\mat\emnist_merge_train_labels.mat'; % Same as train for now!
trainFolder = [dataPath, ''];
testFolder = [dataPath, '']; % Same for now... just on local machine

whichNetwork = 'tutorialNet'; %'pb2';
inputSize = 28;

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
trainAllImgs = GetFileNamesInSubFolders(rootFolder);
testAllImgsCount = numel (trainAllImgs);





% Reduce by sampleFrac and shuffle...
numSampled = round(testAllImgsCount * sampleFrac);
shuffledIndices = randperm(testAllImgsCount, numSampled);
sampledTrainImgs = trainAllImgs(shuffledIndices);



% Get labels.
labels = load(labelKeyTrainPath, 'labelKey');
labels = categorical(labels.labelKey);
labels = labels(shuffledIndices);

% Create table.
tbl = table(sampledTrainImgs, labels);



% split table for train and valid
numTrainImgs = round(numSampled * trainFrac);
trainTbl = tbl(1:numTrainImgs,:); %sampledTrainImgs(1:numTrainImgs);
% trainLabels = trainTbl(:,2:3);
valTbl = tbl(numTrainImgs+1:end,:); %sampledTrainImgs(numTrainImgs+1:end);
% valLabels = valTbl(:,2:3);





%% TEST DATA PREP


% 3) test data: Same as train/val above...
rootFolder = fullfile ( testFolder );
testImgs = GetFileNamesInSubFolders(rootFolder);
testAllImgsCount = numel (testImgs);

assert(~isempty(testImgs), 'No test images found!');

% Reduce by sampleFrac
numTestSampled = round(testAllImgsCount * testFrac);
shuffledIndices = randperm(testAllImgsCount, numTestSampled);
sampledTestImgs = testImgs(shuffledIndices);


% Get labels.
labels = load(labelKeyTestPath, 'labelKey');
labels = categorical(labels.labelKey);
labels = labels(shuffledIndices);


testTbl = table(sampledTestImgs, labels);


fprintf ( 'train -> all (%d), sampled train (%d), valid (%d), test (%d)\n', ...
    testAllImgsCount, ...
    numTrainImgs, ...
    numSampled - numTrainImgs, ...
    numel ( sampledTestImgs ) );
    


%% 2.  set train options
% Temporary local testing to overfit!
numClasses = numel(unique(trainImds.Labels));
maxEpochs = 100; %14;
miniBatchSize = 256;

options = trainingOptions('sgdm', ...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-4, ...
    'MiniBatchSize', miniBatchSize, ...
    'Verbose',1, ...
    'Plots','training-progress');


filterSize = [5 5];
numFilters = 50; 

layers = [ ...
    imageInputLayer([inputSize inputSize 1])
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
close

%%
% Evaluate the trained network on the validation set and calculate
% the validation error.
% predLabels - row = primary key, column = dx or dy
predLabels = classify(trainedNet, valImds);
% % %         valAccuracy = mean(predLabels == valTbl{});
%valError = mean(sqrt((predLabels(:,1) - valTbl{:,2}).^ 2 + (predLabels(:,2) - valTbl{:,3}).^ 2));
        

