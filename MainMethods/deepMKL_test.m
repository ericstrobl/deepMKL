function [pred,acc] = deepMKL_test(xA,y,model,net)
% Deep Multiple Kernel Learning by Span Bound
% 
% Inputs:
% (1) xA = trainng and testing data matrix, where rows are instances and columns are features
% (2) y = testing target vector, where rows are instances
% (3) model = LIBSVM model
% (4) net = net parameters
%
% Outputs:
% (1) pred = predictions
% (2) acc = accuracy
%
% Citation: Strobl EV & Visweswaran S. Deep Multiple Kernel Learning.
% ICMLA, 2013.

dotxA = xA*xA';
[~,Kf] = computeKernels(dotxA,net.sig,net.w,net.nLayers);

[r,~] = size(dotxA);
Ks = reshape(Kf(:,net.nLayers),r,r);
n = net.n;
Ks = Ks(n+1:end,1:n);
[pred, acc, ~] = svmpredict(y, [(1:r-n)' Ks], model);
