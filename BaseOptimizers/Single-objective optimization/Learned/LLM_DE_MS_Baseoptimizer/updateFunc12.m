function [offspring] = updateFunc12(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -1000 * ones(1,D);
    ub = 1000 * ones(1,D);
    eps = 1e-10;
    
    % Normalize constraint violations
    min_cons = min(cons);
    max_cons = max(cons);
    cons_norm = (cons - min_cons) / (max_cons - min_cons + eps);
    
    for i = 1:NP
        % Select four distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        idx = candidates(randperm(NP-1, 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Compute fitness weights
        w1 = abs(popfits(r2)) / (abs(popfits(r2)) + abs(popfits(r3)) + eps);
        w2 = abs(popfits(r4)) / (abs(popfits(r4)) + abs(popfits(r1)) + eps);
        
        % Compute weighted difference vector
        diff_vec = w1 * (popdecs(r2,:) - popdecs(r3,:)) + ...
                   w2 * (popdecs(r4,:) - popdecs(r1,:));
        
        % Compute adaptive scaling factor
        F = 0.5 + 0.5 * tanh(5 * cons_norm(i));
        
        % Generate mutant vector
        mutant = popdecs(r1,:) + F * diff_vec;
        
        % Bound handling by reflection
        for j = 1:D
            if mutant(j) < lb(j)
                mutant(j) = 2*lb(j) - mutant(j);
            elseif mutant(j) > ub(j)
                mutant(j) = 2*ub(j) - mutant(j);
            end
        end
        
        offspring(i,:) = mutant;
    end
end