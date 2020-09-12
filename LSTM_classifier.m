clc
clear
close all

%% create Dataset
number_of_train_data = 100;
for k = 1:number_of_train_data
    r = rand;
    if rand > 0.5
        sample_feature(1,:) = sin(rand(1,20));
        sample_feature(2,:) = cos(rand(1,20));
        XTrain{k} = sample_feature;
        YTrain(k,:) = 1;
    else
        sample_feature(1,:) = cos(rand(1,20));
        sample_feature(2,:) = sin(rand(1,20));
        XTrain{k} = sample_feature;
        YTrain(k,:) = 2;
    end
end
YTrain = categorical(YTrain);

number_of_validation_data = 100;
for k = 1:number_of_validation_data
    r = rand;
    if rand > 0.5
        sample_feature(1,:) = sin(rand(1,20));
        sample_feature(2,:) = cos(rand(1,20));
        XValidation{k} = sample_feature;
        YValidation(k,:) = 1;
    else
        sample_feature(1,:) = cos(rand(1,20));
        sample_feature(2,:) = sin(rand(1,20));
        XValidation{k} = sample_feature;
        YValidation(k,:) = 2;
    end
end
YValidation = categorical(YValidation);

number_of_test_data = 100;
for k = 1:number_of_test_data
    r = rand;
    if rand > 0.5
        sample_feature(1,:) = sin(rand(1,20));
        sample_feature(2,:) = cos(rand(1,20));
        XTest{k} = sample_feature;
        YTest(k,:) = 1;
    else
        sample_feature(1,:) = cos(rand(1,20));
        sample_feature(2,:) = sin(rand(1,20));
        XTest{k} = sample_feature;
        YTest(k,:) = 2;
    end
end
YTest = categorical(YTest);

%% train network
inputSize = size(XTrain{1},1);
numHiddenUnits = 100;
numClasses = 2;

layers = [ sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

maxEpochs = 100;
miniBatchSize = 27;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationData',{XValidation,YValidation}, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);

YPred = classify(net,XTest, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');

acc = sum(YPred == YTest)./numel(YTest);
disp(acc);