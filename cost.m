function [J] = cost(a, y, w, lamb)
    % softmax 
    m = size(a,2);
    maxa = max(a);
    suma = sum(exp(a-maxa));
    softa = exp(a-maxa) ./ suma;
    % regulization
    sumw = 0;
    for l = 1:size(w,2)
        sumw = sumw + sum(sum(w{l}.^2));
    end
    % cross entropy
    J = - sum(sum(y .* log(softa))) / m + 0.5*lamb*sumw;
    return;
end


