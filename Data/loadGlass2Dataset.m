function [Xtrain, Xtest, Ytrain, Ytest] = loadGlass2Dataset()

load Glass2_TestData.mat
load Glass2_TestLabels.mat
load Glass2_TrainData.mat
load Glass2_TrainLabels.mat

Xtrain = zscore(Glass2_TrainData);
Xtest = zscore(Glass2_TestData);
Ytrain = Glass2_TrainLabels;
Ytrain(find(Ytrain==2))=-1;
Ytest = Glass2_TestLabels;
Ytest(find(Ytest==2))=-1;