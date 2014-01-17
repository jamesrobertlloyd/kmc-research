function [ f, df ] = neg_rbf_witness( x, X, Y, ell )
%RBF_WITNESS Summary of this function goes here
%   Detailed explanation goes here
    [f, df] = rbf_witness(x, X, Y, ell);
    f = -f;
    df = -df;
end

