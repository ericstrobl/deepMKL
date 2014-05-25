function [Xtrain, Xtest, Ytrain, Ytest] = loadAudiologyDataset()

load C:\Users\E\Documents\MATLAB\Audiology_Dataset\Audiology_TestData.mat
load C:\Users\E\Documents\MATLAB\Audiology_Dataset\Audiology_TestLabels.mat
load C:\Users\E\Documents\MATLAB\Audiology_Dataset\Audiology_TrainData.mat
load C:\Users\E\Documents\MATLAB\Audiology_Dataset\Audiology_TrainLabels.mat

Xtrain = zscore(Audiology_TrainData);
Xtest = zscore(Audiology_TestData);
Ytrain = Audiology_TrainLabels;
Ytrain(find(Ytrain==0))=-1;
Ytest = Audiology_TestLabels;
Ytest(find(Ytest==0))=-1;