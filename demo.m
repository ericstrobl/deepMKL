clear all
clc

[x, Xtest, y, Ytest] = loadSonarDataset();

%one layer
[model,net] = DeepMKL_train(x,y,1);
[pred,acc] = DeepMKL_test([x;Xtest],Ytest,model,net);

%two layer
[model,net] = DeepMKL_train(x,y,2);
[pred,acc] = DeepMKL_test([x;Xtest],Ytest,model,net);
