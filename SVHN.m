%% RESET
clc; close all; clear;

%% Preparing Data
graphSize = 32;
% Training Set
load train_32x32.mat;
times = 3;
XUp = circshift(X,-1,1);
XDown = circshift(X,1,1);
trainSize = times * size(X,4);
trainData = zeros(graphSize^2, trainSize);
trainLabel = zeros(10, trainSize);
% shift, x3
for idx = 1:trainSize/times
    gray = im2double(rgb2gray(X(:,:,:,idx)));
    grayu = im2double(rgb2gray(XUp(:,:,:,idx)));
    grayd = im2double(rgb2gray(XDown(:,:,:,idx)));
    trainData(:,times*(idx-1)+1:times*idx) = [gray(:),grayu(:),grayd(:)];
    label = y(idx);
    trainLabel(label,times*(idx-1)+1:times*idx) = ones(1,times);
end
% invert color, x2
trainSize = trainSize * 2;
trainData = [trainData, 1-trainData];
trainLabel = [trainLabel, trainLabel];
% Testing Set
load test_32x32.mat;
testSize = size(X,4);
testData = zeros(graphSize^2, testSize);
testLabel = zeros(10, testSize);
for idx = 1:testSize
    gray = rgb2gray(im2double(X(:,:,:,idx)));
    testData(:,idx) = gray(:);
    label = y(idx);
    testLabel(label,idx) = 1;
end
% clear temporary variables
clear gray idx label X y;

%% Desigh parameters
alpha = 0.4;
lamb = 1e-4;
maxEpoch = 300;
batchSize = 500;
batchCount = ceil(trainSize / batchSize);
epochStep = 20;
epochTicks = batchCount*epochStep:batchCount*epochStep:batchCount*maxEpoch;
epochLabels = epochStep:epochStep:maxEpoch;
decayList = [100, 200];

%% Design Network
L = 4;
layer = [graphSize^2, 256, 64, 10];

%% functions
lin = @mylinear;
sig = @sigmoid;
% layer 1 is omitted
% layer L is softmax, implemented in cost and dcost
activation = {sig, sig, lin};

%% initialize weights
w = cell(1,L-1);
for l = 1:L-1
    % Xavier
    bound = sqrt(6.0/(layer(l)+layer(l+1)));
    w{l} = 2 * bound * rand(layer(l+1), layer(l)) - bound;
end

%% Train the Network
J = [];
Acc = [];
% plot the cost
fig = figure;
set(fig, 'position', [100 100 1000 400]);
for iter=1:maxEpoch
    if ismember(iter,decayList)
        alpha = alpha / 2;
    end
    % randomly permute the indexes of samples in training set
    idxs = randperm(trainSize); 
    % for each mini-batch
    for k = 1:batchCount
        % empty initialize
        a = cell(1,L);
        z = cell(1,L);
        delta = cell(1,L);
        % prepare internal inputs in 1st layer denoted by a{1}
        startIdx = (k-1)*batchSize+1;
        endIdx = min(k*batchSize, trainSize);
        realSize = endIdx - startIdx + 1;
        % input
        a{1} = trainData(:,idxs(startIdx:endIdx));
        % prepare labels
        y = trainLabel(:,idxs(startIdx:endIdx));
        % forward computation
        for l=1:L-1
            [a{l+1}, z{l+1}] = fc(w{l}, a{l}, activation{l});
        end
        % Compute delta of last layer
        delta{L} = dcost(a{L},y);
        % backward computation
        for l=L-1:-1:2
            delta{l} = bc(w{l}, z{l}, delta{l+1}, activation{l-1});
        end
        % update weight 
        for l=1:L-1
            % compute the gradient
            grad_w = delta{l+1} * a{l}' + lamb * w{l};
            w{l} = w{l} - alpha*grad_w;
        end 
        % training cost on training batch
        J = [J cost(a{L}, y, w, lamb, batchSize)];
        Acc =[Acc accuracy(a{L}, y)]; 
        % plot training error 
        subplot(1,2,1);
        plot(J);
        xticks(epochTicks);
        xticklabels(epochLabels);
        xlabel('epoch');
        ylabel('Cost');
        % plot training acc
        subplot(1,2,2);
        plot(Acc);
        xticks(epochTicks);
        xticklabels(epochLabels);
        xlabel('epoch');
        ylabel('Accuracy');
        pause(0.000001);
    end
end 
% end training
% plot accuracy
figure
plot(Acc);

%% Test
% test on training set
a = cell(1,L-1);
a{1} = trainData;
for l = 1:L-1
    a{l+1} = fc(w{l}, a{l}, activation{l});
end
trainOut = a{L};
trainAcc = accuracy(a{L}, trainLabel);
fprintf('Accuracy on training dataset is %f%%\n', trainAcc*100);

% test on testing set
a = cell(1,L-1);
a{1} = testData;
for l = 1:L-1
    a{l+1} = fc(w{l}, a{l}, activation{l});
end
testOut = a{L};
testAcc = accuracy(a{L}, testLabel);
fprintf('Accuracy on testing dataset is %f%%\n', testAcc*100);

% other judgement is in PYTHON 'test.py'

%% save
save final.mat w layer J Acc trainOut testOut trainLabel testLabel
