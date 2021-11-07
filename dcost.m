function [dJ] = dcost(a, y)
    % softmax
    m = size(a,2);
    maxa = max(a);
    suma = sum(exp(a-maxa));
    softa = exp(a-maxa) ./ suma;
    % cross entropy
    dJ = (softa - y) / m;
    return;
end
