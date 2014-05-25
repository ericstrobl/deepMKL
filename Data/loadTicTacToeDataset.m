function [Xtrain, Xtest, Ytrain, Ytest] = loadTicTacToeDataset()

load TicTacToe_TestData.mat
load TicTacToe_TestLabels.mat
load TicTacToe_TrainData.mat
load TicTacToe_TrainLabels.mat

Xtrain = zscore(TicTacToe_TrainData);
Xtest = zscore(TicTacToe_TestData);
Ytrain = TicTacToe_TrainLabels';
Ytest = TicTacToe_TestLabels';
Ytrain = Ytrain';
Ytest = Ytest';