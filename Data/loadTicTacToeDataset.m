function [Xtrain, Xtest, Ytrain, Ytest] = loadTicTacToeDataset()

load C:\Users\E\Documents\MATLAB\TicTacToe_Dataset\TicTacToe_TestData.mat
load C:\Users\E\Documents\MATLAB\TicTacToe_Dataset\TicTacToe_TestLabels.mat
load C:\Users\E\Documents\MATLAB\TicTacToe_Dataset\TicTacToe_TrainData.mat
load C:\Users\E\Documents\MATLAB\TicTacToe_Dataset\TicTacToe_TrainLabels.mat

Xtrain = zscore(TicTacToe_TrainData);
Xtest = zscore(TicTacToe_TestData);
Ytrain = TicTacToe_TrainLabels';
Ytest = TicTacToe_TestLabels';
Ytrain = Ytrain';
Ytest = Ytest';