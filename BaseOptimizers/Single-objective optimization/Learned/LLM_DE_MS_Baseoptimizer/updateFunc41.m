function [offspring] = updateFunc41(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    F = 0.8;
    eta = 0.2;
    eps = 1e-10;
    
    for i = 1:NP
        % Select three distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        idx = candidates(randperm(length(candidates), 3));
        r1 = idx(1); r2 = idx(2); r3 = idx(3);
        
        % Fitness-weighted difference
        fit_sum = popfits(r2) + popfits(r1) + eps;
        delta = ((popfits(r2) - popfits(r1)) / fit_sum) .* (popdecs(r2,:) - popdecs(r1,:));
        
        % Constraint-aware perturbation
        cons_perturb = eta * cons(r3) * randn(1, D);
        
        % Mutation
        offspring(i,:) = popdecs(r3,:) + F * delta + cons_perturb;
    end
    
    % Boundary control
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = min(max(offspring, lb), ub);
end