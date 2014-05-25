function [Xtrain, Xtest, Ytrain, Ytest] = loadAudiologyDataset()

load Audiology_TestData.mat
load Audiology_TestLabels.mat
load Audiology_TrainData.mat
load Audiology_TrainLabels.mat

Xtrain = zscore(Audiology_TrainData);
Xtest = zscore(Audiology_TestData);
Ytrain = Audiology_TrainLabels;
Ytrain(find(Ytrain==0))=-1;
Ytest = Audiology_TestLabels;
Ytest(find(Ytest==0))=-1;
