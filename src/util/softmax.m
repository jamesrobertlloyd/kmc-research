function f = softmax( x )
%SOFTMAX Summary of this function goes here
%   Detailed explanation goes here
    e_x = exp(x);
    f = e_x ./ sum(sum(e_x));
end

