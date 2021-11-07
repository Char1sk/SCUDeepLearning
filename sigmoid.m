% x: input value
% f: function value
% df: derivation
function [f,df] = sigmoid(x)
    f = 1 ./ (1 + exp(-x));
    df = f .* (1 - f);
end