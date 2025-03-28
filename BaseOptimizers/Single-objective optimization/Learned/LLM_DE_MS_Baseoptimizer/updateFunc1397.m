% MATLAB Code
function [offspring] = updateFunc1397(popdecs, popfits, cons)
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
    
    % Generate random indices (2 distinct vectors)
    rand_idx = zeros(NP, 2);
    for i = 1:NP
        candidates = setdiff(1:NP, [i, best_idx]);
        rand_idx(i,:) = candidates(randperm(length(candidates), 2));
    end
    
    % Adaptive weights
    w_feas = (1 - cv_norm)/2 + 0.25;
    w_fit = (1 - f_norm)/2 + 0.25;
    w_div = 1 - w_feas - w_fit;
    
    % Adaptive scaling factor
    F = 0.5 * (1 + rand(NP,1)) .* (1 - cv_norm);
    
    % Mutation vectors
    diff_feas = bsxfun(@minus, x_feas, popdecs);
    diff_best = bsxfun(@minus, x_best, popdecs);
    diff_rand = popdecs(rand_idx(:,1),:) - popdecs(rand_idx(:,2),:);
    
    mutants = popdecs + F .* (w_feas .* diff_feas + w_fit .* diff_best + w_div .* diff_rand);
    
    % Boundary handling with adaptive reflection
    reflect_scale = 0.1 + 0.4 * rand(NP, 1);
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
    
    % Enhanced exploration for highly constrained solutions
    high_cv = cv_norm > 0.8;
    if any(high_cv)
        offspring(high_cv,:) = bsxfun(@plus, x_feas, ...
                               0.5 * bsxfun(@times, (ub - lb), rand(sum(high_cv),D)-0.5));
    end
end