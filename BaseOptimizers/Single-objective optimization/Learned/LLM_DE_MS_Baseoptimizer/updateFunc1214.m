% MATLAB Code
function [offspring] = updateFunc1214(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Fitness-weighted base vector
    f_min = min(popfits);
    weights = 1./(popfits - f_min + eps);
    x_base = sum(bsxfun(@times, popdecs, weights), 1) ./ sum(weights);
    
    % 2. Constraint-adaptive scaling factor
    cv_abs = abs(cons);
    max_cv = max(cv_abs);
    F = 0.3 + 0.5 * (1 - cv_abs/(max_cv + eps));
    
    % 3. Directional mutation
    idx = (1:NP)';
    r1 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r2 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r3 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r4 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    mutants = bsxfun(@plus, x_base, bsxfun(@times, F, diff1) + 0.5*bsxfun(@times, F, diff2));
    
    % 4. Elite-guided crossover
    CR = 0.1 + 0.7 * (1 - cv_abs/(max_cv + eps));
    mask = rand(NP, D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 5. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end