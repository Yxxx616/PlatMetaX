% MATLAB Code
function [offspring] = updateFunc1373(popdecs, popfits, cons)
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
    
    % Find feasible solutions and best solution
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        feasible_center = mean(popdecs(feasible_mask, :), 1);
    else
        [~, min_cv_idx] = min(cons);
        feasible_center = popdecs(min_cv_idx, :);
    end
    
    [~, best_idx] = min(popfits + 1e6 * cv_pos);
    x_best = popdecs(best_idx, :);
    
    % Generate random indices for mutation
    rand_idx = zeros(NP, 2);
    for i = 1:NP
        available = setdiff(1:NP, [i, best_idx]);
        rand_idx(i,:) = available(randperm(length(available), 2));
    end
    
    % Adaptive scaling factors
    F = 0.5 * (1 - cv_norm);
    G = 0.3 * (1 - f_norm);
    
    % Mutation components
    diff_elite = bsxfun(@minus, x_best, popdecs);
    diff_rand = popdecs(rand_idx(:,1),:) - popdecs(rand_idx(:,2),:);
    diff_feas = bsxfun(@minus, feasible_center, popdecs);
    
    % Combined mutation
    v_elite = popdecs + bsxfun(@times, F, diff_elite) + bsxfun(@times, G, diff_rand);
    v_feas = popdecs + bsxfun(@times, F, diff_feas);
    
    % Adaptive weights
    w = 0.5 * (1 - cv_norm);
    mutants = bsxfun(@times, w, v_elite) + bsxfun(@times, 1-w, v_feas);
    
    % Constraint-aware crossover
    CR = 0.9 * (1 - cv_norm) + 0.1;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
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
        offspring(out_of_bounds,:) = bsxfun(@plus, feasible_center, ...
                                   0.1*bsxfun(@times, (ub - lb), rand(sum(out_of_bounds),D)-0.5));
    end
end