function [model,net] = deepMKL_train(x,y,nLayers,C,LR,maxI)
% Deep Multiple Kernel Learning by Span Bound
% 
% Inputs:
% (1) x = trainng data matrix, where rows are instances and columns are features
% (2) y = training target vector, where rows are instances
% (3) nLayers = number of layers, 1 or 2
% (4) C = SVM penalty constant (default=10)
% (5) LR = learning rate (default=1E-4)
% (6) maxI = maximum number of iterations (default=100)
%
% Outputs:
% (1) model = LIBSVM model
% (2) net = net parameters
%
% Citation: Strobl EV & Visweswaran S. Deep Multiple Kernel Learning.
% ICMLA, 2013.


%default values
SetDefaultValue(4,'C',10);
SetDefaultValue(5,'LR',1E-4);
SetDefaultValue(6,'maxI',100);


%initialize weights
betas = ones(nLayers,4)./4;

%initialize kernels
dotx = x*x';
sig = DetermineSig(dotx);
[~,Kf] = computeKernels(dotx,sig,betas,nLayers);

%alternating opt
[r,~] = size(x);
span = 0;
for t=1:maxI,
   
    %train SVM
    Ks = reshape(Kf(:,nLayers),r,r);
    model = svmtrain(y, [(1:r)',Ks], ['-c ' num2str(C) ' -t 4 -q 1']);
    
    %kernels
    [K,Kf] = computeKernels(dotx,sig,betas,nLayers);
    
    %span gradient
    if nLayers==1,
        [betas,spanT] = grad1Layer(model,betas,LR,Kf,K,y);
    elseif nLayers==2,
        [betas,spanT] = grad2Layer(model,betas,LR,Kf,K,sig,y);
    elseif nLayers==3,
        [betas,spanT] = grad3Layer(model,betas,LR,Kf,K,sig,y);
    end

    %feasible region projection
    betas(find(betas<0))=0; %non-negative
    betas = betas./repmat(sum(betas,2),1,4); %sum to 1
    
    %stopping conditions
    if isnan(sum(betas)),
        error('myApp:argChk', 'Learning rate is too high');
    elseif abs(span-spanT)<1E-4 && t>5,
        break;
    end
    span=spanT;
    
end

%final model
net.w = betas;
net.sig = sig;
net.nLayers = nLayers;
net.n = r;
