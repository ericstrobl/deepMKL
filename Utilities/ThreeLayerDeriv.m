function answ = ThreeLayerDeriv(Kf2,K2,Kf1,K1,betas2,betas3,sig)
    [r,~] = size(K2); r=sqrt(r);
    Kf2 = reshape(Kf2,r,r); 
    K2 = reshape(K2,r,r,4); 
    
    answ = betas3(1).*normalizeKernel_Grad(Kf2,rbfDeriv(K2(:,:,1),Kf1,K1,betas2,sig)) + betas3(2)*normalizeKernel_Grad(Kf2,polyDeriv(K2(:,:,2),Kf1,K1,betas2,sig))...
        + betas3(3)*normalizeKernel_Grad(Kf2,poly2Deriv(K2(:,:,3),Kf1,K1,betas2,sig)) + betas3(4)*normalizeKernel_Grad(Kf2,linDeriv(Kf1,K1,betas2,sig));
    
end

function answ = rbfDeriv(K2,Kf1,K1,betas2,sig)
    answ = exp(-2/(2*sig^2).*(1-K2)).*(2/(2*sig^2).*TwoLayerDeriv(Kf1,K1,betas2,sig));
end

function answ = polyDeriv(K2,Kf1,K1,betas2,sig)
    answ = 2.*(K2+1).*TwoLayerDeriv(Kf1,K1,betas2,sig);
end

function answ = poly2Deriv(K2,Kf1,K1,betas2,sig)
    answ = 3.*((K2+1).^2).*TwoLayerDeriv(Kf1,K1,betas2,sig);
end

function answ = linDeriv(Kf1,K1,betas2,sig)
    answ = TwoLayerDeriv(Kf1,K1,betas2,sig);
end

function answ = normalizeKernel_Grad(Kf,KfDeriv)
dKf = diag(Kf);
dKfDeriv = diag(KfDeriv);
answ = KfDeriv.*((dKf*dKf').^0.5)...
    +Kf.*(-0.5.*(dKf*dKf').^(1.5))...
    .*(dKfDeriv*dKf')+(dKf*dKfDeriv');
end
