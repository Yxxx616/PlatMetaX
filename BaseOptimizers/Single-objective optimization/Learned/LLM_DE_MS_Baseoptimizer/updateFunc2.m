function [offspring] = updateFunc2(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    epsilon = 1e-6;
    max_cons = max(cons);
    sum_cons = sum(cons);
    
    for i = 1:NP
        % Select three distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        r = candidates(randperm(NP-1, 3));
        r1 = r(1); r2 = r(2); r3 = r(3);
        
        % Fitness-weighted difference vector
        fit_weights = popfits(r2) / (popfits(r2) + popfits(r3) + epsilon);
        d = fit_weights .* (popdecs(r2,:) - popdecs(r3,:));
        
        % Constraint-aware scaling factor
        F = 0.5 + 0.5 * (cons(i) / (max_cons + 1));
        
        % Constraint-based perturbation
        p = (cons(i) / (sum_cons + 1)) * randn(1, D);
        
        % Generate mutant vector
        offspring(i,:) = popdecs(r1,:) + F .* d + p;
    end
    
    % Ensure bounds are maintained (assuming [0,10] range)
    offspring = min(max(offspring, 0), 10);
end