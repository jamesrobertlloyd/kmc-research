function [ f, df ] = rbf_witness( x, X, Y, ell )
%RBF_WITNESS Summary of this function goes here
%   Detailed explanation goes here
    m = size(X, 1);
    n = size(Y, 1);
    K1 = rbf_dot(x', X, ell);
    K2 = rbf_dot(x', Y, ell);
    f = sum(K1, 2)/ m - sum(K2, 2)/ n;
    df = -(1 ./ (ell^2)) .* (...
         x*f + ...
         (K2*Y)'/n - ...
         (K1*X)'/m);
end

