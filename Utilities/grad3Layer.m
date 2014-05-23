function [betas,span] = grad3Layer(model,betas,LR,Kf,K,sig,y)

%third layer
dTdT9 = SpanBoundDeriv(model,Kf(:,3),K(:,1,3),y);
dTdT10 = SpanBoundDeriv(model,Kf(:,3),K(:,2,3),y);
dTdT11 = SpanBoundDeriv(model,Kf(:,3),K(:,3,3),y);
[dTdT12,span] = SpanBoundDeriv(model,Kf(:,3),K(:,4,3),y);

%second layer
Kd5 = TwoLayerDeriv(Kf(:,2),K(:,1,2),betas(3,:),sig);
Kd6 = TwoLayerDeriv(Kf(:,2),K(:,2,2),betas(3,:),sig);
Kd7 = TwoLayerDeriv(Kf(:,2),K(:,3,2),betas(3,:),sig);
Kd8 = TwoLayerDeriv(Kf(:,2),K(:,4,2),betas(3,:),sig);
dTdT5 = SpanBoundDeriv(model,Kf(:,3),Kd5,y);
dTdT6 = SpanBoundDeriv(model,Kf(:,3),Kd6,y);
dTdT7 = SpanBoundDeriv(model,Kf(:,3),Kd7,y);
dTdT8 = SpanBoundDeriv(model,Kf(:,3),Kd8,y);

%first layer
Kd1 = ThreeLayerDeriv(Kf(:,2),K(:,:,2),Kf(:,1),K(:,1,1),betas(2,:),betas(3,:),sig);
Kd2 = ThreeLayerDeriv(Kf(:,2),K(:,:,2),Kf(:,1),K(:,2,1),betas(2,:),betas(3,:),sig);
Kd3 = ThreeLayerDeriv(Kf(:,2),K(:,:,2),Kf(:,1),K(:,3,1),betas(2,:),betas(3,:),sig);
Kd4 = ThreeLayerDeriv(Kf(:,2),K(:,:,2),Kf(:,1),K(:,4,1),betas(2,:),betas(3,:),sig);
dTdT1 = SpanBoundDeriv(model,Kf(:,3),Kd1,y);
dTdT2 = SpanBoundDeriv(model,Kf(:,3),Kd2,y);
dTdT3 = SpanBoundDeriv(model,Kf(:,3),Kd3,y);
dTdT4 = SpanBoundDeriv(model,Kf(:,3),Kd4,y);

%display
disp(['Span: ' num2str(span)])

%gradient step
Dbetas = [dTdT1 dTdT2 dTdT3 dTdT4; dTdT5 dTdT6 dTdT7 dTdT8;...
dTdT9 dTdT10 dTdT11 dTdT12];
betas = betas - LR.*Dbetas;

end
