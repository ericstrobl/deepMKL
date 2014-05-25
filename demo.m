clear all
clc

%% Trains 1,2 and 3 layer models on three example datasets

%% GERMAN
disp('GERMAN')

[x, Xtest, y, Ytest] = loadGermanDatasetCP();

%one layer
[model,net] = deepMKL_train(x,y,1,1E-2);
[pred,acc] = deepMKL_test([x;Xtest],Ytest,model,net);

%two layer
[model,net] = deepMKL_train(x,y,2,1E-5);
[pred,acc] = deepMKL_test([x;Xtest],Ytest,model,net);

%three layer
[model,net] = deepMKL_train(x,y,3,1E-4);
[pred,acc] = deepMKL_test([x;Xtest],Ytest,model,net);

%% SONAR
disp('SONAR')
[x, Xtest, y, Ytest] = loadSonarDatasetCP();
 
%one layer
[model,net] = deepMKL_train(x,y,1,1E-2);
[pred,acc] = deepMKL_test([x;Xtest],Ytest,model,net);

%two layer
[model,net] = deepMKL_train(x,y,2,1E-2);
[pred,acc] = deepMKL_test([x;Xtest],Ytest,model,net);

%three layer
[model,net] = deepMKL_train(x,y,3,1E-2);
[pred,acc] = deepMKL_test([x;Xtest],Ytest,model,net);

%% AUDIOLOGY
disp('AUSTRALIAN')
[x, Xtest, y, Ytest] = loadAustralianDatasetCP();
 
%one layer
[model,net] = deepMKL_train(x,y,1,1E-2);
[pred,acc] = deepMKL_test([x;Xtest],Ytest,model,net);

%two layer
[model,net] = deepMKL_train(x,y,2,1E-3);
[pred,acc] = deepMKL_test([x;Xtest],Ytest,model,net);

%three layer
[model,net] = deepMKL_train(x,y,3,1E-4);
[pred,acc] = deepMKL_test([x;Xtest],Ytest,model,net);
