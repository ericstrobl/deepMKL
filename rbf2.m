function K = rbf2(coord,sig)
K = exp(-2/(2*sig^2).*(1-coord));
