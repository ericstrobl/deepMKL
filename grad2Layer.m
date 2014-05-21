function [betas,span] = grad2Layer(model,betas,LR,Kf,K,sig,y)

    %second layer
    dTdT5 = SpanBoundDeriv(model,Kf(:,2),K(:,1,2),y);
    dTdT6 = SpanBoundDeriv(model,Kf(:,2),K(:,2,2),y);
    dTdT7 = SpanBoundDeriv(model,Kf(:,2),K(:,3,2),y);
    [dTdT8,span] = SpanBoundDeriv(model,Kf(:,2),K(:,4,2),y);

    %first layer
    Kd1 = TwoLayerDeriv(Kf(:,1),K(:,1,1),betas(2,:),sig);
    Kd2 = TwoLayerDeriv(Kf(:,1),K(:,2,1),betas(2,:),sig);
    Kd3 = TwoLayerDeriv(Kf(:,1),K(:,3,1),betas(2,:),sig);
    Kd4 = TwoLayerDeriv(Kf(:,1),K(:,4,1),betas(2,:),sig);
    dTdT1 = SpanBoundDeriv(model,Kf(:,2),Kd1,y);
    dTdT2 = SpanBoundDeriv(model,Kf(:,2),Kd2,y);
    dTdT3 = SpanBoundDeriv(model,Kf(:,2),Kd3,y);
    dTdT4 = SpanBoundDeriv(model,Kf(:,2),Kd4,y);

    %display
    disp(['Span: ' num2str(span)])
    
    %gradient step
    Dbetas = [dTdT1 dTdT2 dTdT3 dTdT4; dTdT5 dTdT6 dTdT7 dTdT8];
    betas = betas - repmat(LR,2,4).*Dbetas;
end
