function [betas,span] = grad1Layer(model,betas,LR,Kf,K,y)

    %first layer
    dTdT1 = SpanBoundDeriv(model,Kf(:,1),K(:,1,1),y);
    dTdT2 = SpanBoundDeriv(model,Kf(:,1),K(:,2,1),y);
    dTdT3 = SpanBoundDeriv(model,Kf(:,1),K(:,3,1),y);
    [dTdT4,span] = SpanBoundDeriv(model,Kf(:,1),K(:,4,1),y);

    %display
    disp(['Span: ' num2str(span)])
    
    %gradient step
    Dbetas = [dTdT1 dTdT2 dTdT3 dTdT4];
    betas = betas - LR.*Dbetas;
    
end
