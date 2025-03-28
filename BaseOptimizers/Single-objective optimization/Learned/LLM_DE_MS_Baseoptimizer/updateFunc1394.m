% MATLAB Code
function [offspring] = updateFunc1394(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    eps = 1e-12;
    
    % Normalize constraints and fitness
    cv_pos = max(0, cons);
    cv_min = min(cv_pos);
    cv_max = max(cv_pos) + eps;
    cv_norm = (cv_pos - cv_min) / (cv_max - cv_min);
    
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    f_norm = (popfits - f_min) / f_range;
    
    % Find reference points
    penalty = popfits + 1e6 * cv_pos;
    [~, best_idx] = min(penalty);
    x_best = popdecs(best_idx, :);
    
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        x_feas = mean(popdecs(feasible_mask, :), 1);
    else
        [~, min_cv_idx] = min(cons);
        x_feas = popdecs(min_cv_idx, :);
    end
    
    % Generate random indices (4 distinct vectors)
    rand_idx = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(1:NP, [i, best_idx]);
        rand_idx(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % Adaptive weights
    w_best = (1 - cv_norm)/3 + (1 - f_norm)/3;
    w_feas = (1 - cv_norm)/3 + 1/3;
    w_div = 1 - w_best - w_feas;
    
    % Adaptive scaling factor
    F = 0.5 * (0.3 + 0.7 * (1 - cv_norm));
    
    % Mutation
    diff_best = x_best - popdecs;
    diff_feas = x_feas - popdecs;
    diff_rand1 = popdecs(rand_idx(:,1),:) - popdecs(rand_idx(:,2),:);
    diff_rand2 = popdecs(rand_idx(:,3),:) - popdecs(rand_idx(:,4),:);
    
    mutants = popdecs + bsxfun(@times, F .* w_best, diff_best) + ...
              bsxfun(@times, F .* w_feas, diff_feas) + ...
              bsxfun(@times, F .* w_div, diff_rand1 + diff_rand2);
    
    % Boundary handling with adaptive reflection
    reflect_scale = 0.2 + 0.3 * rand(NP, 1);
    for j = 1:D
        below = mutants(:,j) < lb(j);
        above = mutants(:,j) > ub(j);
        
        mutants(below,j) = lb(j) + reflect_scale(below) .* (lb(j) - mutants(below,j));
        mutants(above,j) = ub(j) - reflect_scale(above) .* (mutants(above,j) - ub(j));
        
        % Final clamping
        mutants(:,j) = min(max(mutants(:,j), lb(j)), ub(j));
    end
    
    % Adaptive crossover
    CR = 0.85 * (1 - cv_norm) + 0.15;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Diversity preservation for solutions with high constraint violation
    high_cv = cv_norm > 0.8;
    if any(high_cv)
        offspring(high_cv,:) = bsxfun(@plus, x_feas, ...
                               0.2 * bsxfun(@times, (ub - lb), rand(sum(high_cv),D)-0.5));
    end
end