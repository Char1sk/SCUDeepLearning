function [a_next, z_next] = fc(w, a, func)
    % forward computing 
    z_next = w * a;
    [a_next,~] = func(z_next);
    return;
end
