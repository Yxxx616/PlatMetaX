% MATLAB Code
function [offspring] = updateFunc1219(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps_val = 1e-10;
    
    % 1. Calculate fitness-weighted centroid
    weights = 1./(abs(popfits) + abs(cons) + eps_val);
    x_wbest = sum(popdecs .* weights, 1) ./ sum(weights);
    
    % 2. Adaptive scaling factor with constraint awareness
    f_min = min(popfits);
    f_max = max(popfits);
    cv_abs = abs(cons);
    max_cv = max(cv_abs);
    
    F = 0.5 * (1 - cv_abs./(max_cv + eps_val)) .* ...
        (1 + (popfits - f_min)./(f_max - f_min + eps_val));
    F = F(:, ones(1, D));
    
    % 3. Constraint-aware mutation
    idx = (1:NP)';
    r1 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r2 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r3 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r4 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    cv_factor = cv_abs(r3)./(max_cv + eps_val);
    mutants = x_wbest(ones(NP,1), :) + F .* diff1 + ...
              F .* diff2 .* cv_factor(:, ones(1, D));
    
    % 4. Dynamic crossover rate
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.9 - 0.6 * (ranks/NP) .* (cv_abs/(max_cv + eps_val));
    CR = CR(:, ones(1, D));
    
    % 5. Crossover
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 6. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end