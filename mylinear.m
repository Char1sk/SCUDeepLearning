% x: input value
% f: function value
% df: derivation
function [f,df] = mylinear(x)
    f = x;
    df = ones(size(x));
end