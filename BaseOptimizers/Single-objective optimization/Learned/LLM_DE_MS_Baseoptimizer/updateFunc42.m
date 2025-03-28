function [offspring] = updateFunc42(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    F = 0.7;
    eta = 0.15;
    eps = 1e-10;
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    for i = 1:NP
        % Select four distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Fitness-weighted difference
        fit_denom = abs(popfits(r2)) + abs(popfits(r1)) + eps;
        delta = ((popfits(r2) - popfits(r1)) / fit_denom .* ...
                (popdecs(r2,:) - popdecs(r1,:));
        
        % Constraint-aware perturbation using tanh smoothing
        cons_weight = eta * tanh(abs(cons(r3)));
        perturb = cons_weight .* (popdecs(r4,:) - popdecs(r3,:));
        
        % Mutation
        offspring(i,:) = popdecs(r3,:) + F * delta + perturb;
    end
    
    % Boundary control
    offspring = min(max(offspring, lb), ub);
end