% MATLAB Code
function [offspring] = updateFunc1375(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    eps = 1e-12;
    
    % Normalize constraints and fitness
    cv_pos = max(0, cons);
    cv_max = max(cv_pos) + eps;
    cv_norm = cv_pos / cv_max;
    
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    f_norm = (popfits - f_min) / f_range;
    
    % Find reference points
    [~, best_idx] = min(popfits + 1e6 * cv_pos);
    x_best = popdecs(best_idx, :);
    
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        x_feas = mean(popdecs(feasible_mask, :), 1);
    else
        [~, min_cv_idx] = min(cons);
        x_feas = popdecs(min_cv_idx, :);
    end
    
    % Generate random indices for differential vectors
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    for i = 1:NP
        while rand_idx1(i) == i || rand_idx1(i) == best_idx
            rand_idx1(i) = randi(NP);
        end
        while rand_idx2(i) == i || rand_idx2(i) == best_idx || rand_idx2(i) == rand_idx1(i)
            rand_idx2(i) = randi(NP);
        end
    end
    
    % Adaptive scaling factors
    F1 = 0.8 * (1 - cv_norm);
    F2 = 0.6 * (1 - f_norm);
    
    % Mutation
    diff_best = bsxfun(@minus, x_best, popdecs);
    diff_rand = popdecs(rand_idx1,:) - popdecs(rand_idx2,:);
    diff_feas = bsxfun(@minus, x_feas, popdecs);
    
    mutants = popdecs + bsxfun(@times, F1, diff_best) + bsxfun(@times, F2, diff_rand) + ...
              0.4 * bsxfun(@times, cv_norm, diff_feas);
    
    % Crossover
    CR = 0.9 * (1 - cv_norm) + 0.1;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    for j = 1:D
        below = offspring(:,j) < lb(j);
        above = offspring(:,j) > ub(j);
        
        offspring(below,j) = 2*lb(j) - offspring(below,j);
        offspring(above,j) = 2*ub(j) - offspring(above,j);
        
        % Ensure within bounds
        offspring(below,j) = max(min(offspring(below,j), ub(j)), lb(j));
        offspring(above,j) = min(max(offspring(above,j), ub(j)), lb(j));
    end
    
    % Final feasibility enforcement
    out_of_bounds = any(bsxfun(@lt, offspring, lb) | bsxfun(@gt, offspring, ub), 2);
    if any(out_of_bounds)
        offspring(out_of_bounds,:) = bsxfun(@plus, x_feas, ...
                                   0.1*bsxfun(@times, (ub - lb), rand(sum(out_of_bounds),D)-0.5));
    end
end