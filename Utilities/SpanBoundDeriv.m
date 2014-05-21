function [dTdT,Span] = SpanBoundDeriv(model,Kn,Kd,Ytrain)

[ri,~] = size(Kn); ri = sqrt(ri);
Kn = reshape(Kn,ri,ri);
Kd = reshape(Kd,ri,ri);

[sv_indicesS,IX] = sort(model.sv_indices);
sv_coefS = abs(model.sv_coef(IX));

%KthetaSV
Ksv = Kn(sv_indicesS, sv_indicesS);
[r,~] = size(Ksv);
Ksv = Ksv+0.001.*(eye(r));% % 
KTsv = ones(r+1,r+1);
KTsv(1:end-1,1:end-1) = Ksv;
KTsv(end,end) = 0;
KTsv(end,end) = 0.001; %%

%KzeroSV
Kksv = Kd(sv_indicesS, sv_indicesS);
Kk0sv = zeros(r+1,r+1);
Kk0sv(1:end-1,1:end-1) = Kksv;

%Q
Q = zeros(r+1,r+1);
Q(1:(r+1):end) = [1./abs(sv_coefS); 0];

%G
G = zeros(r+1,r+1);
G(1:(r+1):end) = [-1./(abs(sv_coefS)).^2; 0];

%B
B = KTsv+Q;
Binv = inv(B);

%A
A = inv(KTsv);
A(end,:) = [];
A(:,end) = [];

%Ysv
Ysv = zeros(r,r);
Ysv(1:(r+1):end) = Ytrain(sv_indicesS);

%F
F = diag([Ysv*A*Kksv*Ysv*sv_coefS; 0]);

%dSdT
dSdT = (1/diag(Binv)).^2*diag((Binv*(Kk0sv+G*F)*Binv))-diag(G*F);

%dTdT
c=5;
d=0;
S = (1./diag(B))-diag(Q);
dTdT = sum((1./(1+exp(-c*S+d)).^2).*exp(-c*S+d).*(-c*dSdT));

%Span
Span = sum(1./(1+exp(-c*S+d)));
