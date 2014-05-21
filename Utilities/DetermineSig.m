function sig = DetermineSig(dot)
Dis = PairWiseDistance(dot);
xDis = triu(Dis);
xDis = xDis(:);
xDis(xDis==0)=[];
sig = median(xDis);
if isnan(sig),
    sig = 1;
end
end
