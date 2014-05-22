function kNorm = normalizeKernel(K)
kNorm = K./sqrt(diag(K)*diag(K)');
