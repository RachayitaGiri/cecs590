clear;
clc;

net = alexnet;

layers = net.Layers;
layers 

rootFolder = 'cifar10Train';
categories = {'Deer','Dog','Frog','Cat'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @readFunctionTrain;
% Change the number 50 to as many training images as you would like to use
% how does increasing the number of images change the 
% accuracy of the classifier?
[trainingSet, ~] = splitEachLabel(imds, 50, 'randomize'); 



layers = layers(1:end-3);
layers(end+1) = fullyConnectedLayer(64, 'Name', 'special_2');
layers(end+1) = reluLayer;
layers(end+1) = fullyConnectedLayer(4, 'Name', 'fc8_2 ');
layers(end+1) = softmaxLayer;
layers(end+1) = classificationLayer();

layers(end-2).WeightLearnRateFactor = 10;
layers(end-2).WeightL2Factor = 1;
layers(end-2).BiasLearnRateFactor = 20;
layers(end-2).BiasL2Factor = 0;

opts = trainingOptions('sgdm', ...
    'LearnRateSchedule', 'none',...
    'InitialLearnRate', .0001,... 
    'MaxEpochs', 20, ...
    'MiniBatchSize', 128);

convnet = trainNetwork(imds, layers, opts);

rootFolder = 'cifar10Test';
testDS = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
testDS.ReadFcn = @readFunctionTrain;

[labels,err_test] = classify(convnet, testDS, 'MiniBatchSize', 64);

confMat = confusionmat(testDS.Labels, labels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))