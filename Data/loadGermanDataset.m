function [Xtrain, Xtest, Ytrain, Ytest] = loadGermanDataset()

load German_TestData.mat
load German_TestLabels.mat
load German_TrainData.mat
load German_TrainLabels.mat

Xtrain = zscore(German_TrainData);
Xtest = zscore(German_TestData);
Ytrain = German_TrainLabels;
Ytrain(find(Ytrain==2))=-1;
Ytest = German_TestLabels;
Ytest(find(Ytest==2))=-1;
