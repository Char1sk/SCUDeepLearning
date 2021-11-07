function delta = bc(w, z, delta_next, func)
    [~,df] = func(z);
    delta = (w' * delta_next) .* df;
    return;
end
