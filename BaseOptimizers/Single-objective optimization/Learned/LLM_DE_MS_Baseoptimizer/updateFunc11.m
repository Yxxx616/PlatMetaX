function [offspring] = updateFunc11(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Normalize constraint violations to [0,1]
    cons_norm = abs(cons) / max(abs(cons));
    
    for i = 1:NP
        % Select three distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        idx = candidates(randperm(NP-1, 3));
        r1 = idx(1); r2 = idx(2); r3 = idx(3);
        
        % Fitness-based difference vector
        fit_sum = abs(popfits(r2)) + abs(popfits(r3));
        if fit_sum > 1e-10
            w = abs(popfits(r2)) / fit_sum;
        else
            w = 0.5;
        end
        diff_vec = w * (popdecs(r2,:) - popdecs(r3,:));
        
        % Constraint-aware scaling factor
        F = 0.5 * (1 + cons_norm(i));
        
        % Generate mutant vector
        mutant = popdecs(r1,:) + F * diff_vec;
        
        % Bound handling by reflection
        lb = -1000 * ones(1,D);
        ub = 1000 * ones(1,D);
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