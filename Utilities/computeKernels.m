function [K,Kf] = computeKernels(dotx,sig,betas,nLayers)

[r,~] = size(dotx);
K = zeros(r*r,4,nLayers);
Kf = zeros(r*r,nLayers);

for t=1:nLayers,
    if t==1,
        Krbf = rbf(dotx,sig);
    else
        Krbf = normalizeKernel(rbf2(dotx,sig));
    end
    Kpoly2 = normalizeKernel((dotx+1).^2);
    Kpoly3 = normalizeKernel((dotx+1).^3);
    Klin = normalizeKernel(dotx);
    
    K(:,1,t) = Krbf(:);
    K(:,2,t) = Kpoly2(:);
    K(:,3,t) = Kpoly3(:);
    K(:,4,t) = Klin(:);

    dotx = K(:,:,t).*repmat(betas(t,:),r*r,1);
    Kf(:,t) = sum(dotx,2);
    dotx = reshape(Kf(:,t),r,r);
end

