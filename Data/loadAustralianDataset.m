function [Xtrain, Xtest, Ytrain, Ytest] = loadAustralianDataset()

load Australian_TestData.mat
load Australian_TestLabels.mat
load Australian_TrainData.mat
load Australian_TrainLabels.mat

Xtrain = zscore(Australian_TrainData);
Xtest = zscore(Australian_TestData);
Ytrain = Australian_TrainLabels;
Ytrain(find(Ytrain==0))=-1;
Ytest = Australian_TestLabels;
Ytest(find(Ytest==0))=-1;