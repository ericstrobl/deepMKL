function [model,net] = deepMKL_train(x,y,nLayers,LR,maxI,C)
% Deep Multiple Kernel Learning by Span Bound
% 
% Inputs:
% (1) x = trainng data matrix, where rows are instances and columns are features
% (2) y = training target vector, where rows are instances
% (3) nLayers = number of layers, 1, 2 or 3
% (4) LR = learning rate (default=1E-4)
% (5) maxI = maximum number of iterations (default=100)
% (6) C = SVM penalty constant (default=10)
%
% Outputs:
% (1) model = LIBSVM model
% (2) net = net parameters
%
% Citation: Strobl EV & Visweswaran S. Deep Multiple Kernel Learning.
% ICMLA, 2013.


%default values
SetDefaultValue(4,'LR',1E-4);
SetDefaultValue(5,'maxI',100);
SetDefaultValue(6,'C',10);


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
    if sum(betas(end,:))>1,
        betas(end,:) = betas(end,:)./sum(betas(end,:)); %trace final layer upper bound
    end
    
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
