function [offspring] = updateFunc44(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    eta = 0.3; % Constraint weight
    alpha = 0.1; % Diversity factor
    eps = 1e-10; % Small constant
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Precompute min and max fitness
    f_min = min(popfits);
    f_max = max(popfits);
    
    for i = 1:NP
        % Select four distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Fitness-guided direction
        fit_denom = abs(popfits(r2)) + abs(popfits(r1)) + eps;
        delta = ((popfits(r2) - popfits(r1)) / fit_denom) .* ...
                (popdecs(r2,:) - popdecs(r1,:));
        
        % Adaptive scaling factor
        F = 0.5 * (1 + (popfits(r3) - f_min) / (f_max - f_min + eps));
        
        % Constraint-aware perturbation
        cons_weight = eta * tanh(abs(cons(r3)));
        perturb = cons_weight .* (popdecs(r4,:) - popdecs(r3,:));
        
        % Diversity component
        rand_comp = alpha * randn(1,D) .* (ub - lb);
        
        % Mutation
        offspring(i,:) = popdecs(r3,:) + F * delta + perturb + rand_comp;
    end
    
    % Boundary control with reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring = offspring .* (~mask_lb & ~mask_ub) + ...
                (2*lb - offspring) .* mask_lb + ...
                (2*ub - offspring) .* mask_ub;
    
    % Additional check to ensure no NaN values
    nan_mask = isnan(offspring);
    if any(nan_mask(:))
        offspring(nan_mask) = lb(nan_mask) + rand(sum(nan_mask(:)),1) .* ...
                            (ub(nan_mask) - lb(nan_mask));
    end
end