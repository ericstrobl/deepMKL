function [Xtrain, Xtest, Ytrain, Ytest] = loadAustralianDataset()

load C:\Users\E\Documents\MATLAB\Australian_Dataset\Australian_TestData.mat
load C:\Users\E\Documents\MATLAB\Australian_Dataset\Australian_TestLabels.mat
load C:\Users\E\Documents\MATLAB\Australian_Dataset\Australian_TrainData.mat
load C:\Users\E\Documents\MATLAB\Australian_Dataset\Australian_TrainLabels.mat

Xtrain = zscore(Australian_TrainData);
Xtest = zscore(Australian_TestData);
Ytrain = Australian_TrainLabels;
Ytrain(find(Ytrain==0))=-1;
Ytest = Australian_TestLabels;
Ytest(find(Ytest==0))=-1;