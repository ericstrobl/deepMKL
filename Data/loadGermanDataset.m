function [Xtrain, Xtest, Ytrain, Ytest] = loadGermanDataset()

load C:\Users\E\Documents\MATLAB\German_Dataset\German_TestData.mat
load C:\Users\E\Documents\MATLAB\German_Dataset\German_TestLabels.mat
load C:\Users\E\Documents\MATLAB\German_Dataset\German_TrainData.mat
load C:\Users\E\Documents\MATLAB\German_Dataset\German_TrainLabels.mat

Xtrain = zscore(German_TrainData);
Xtest = zscore(German_TestData);
Ytrain = German_TrainLabels;
Ytrain(find(Ytrain==2))=-1;
Ytest = German_TestLabels;
Ytest(find(Ytest==2))=-1;