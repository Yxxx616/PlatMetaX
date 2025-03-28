% MATLAB Code
function [offspring] = updateFunc1367(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    eps = 1e-12;
    
    % Normalize constraints (0 to 1)
    cv_pos = max(0, cons);
    cv_max = max(cv_pos) + eps;
    cv_norm = cv_pos / cv_max;
    
    % Normalize fitness (0 to 1)
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    f_norm = (popfits - f_min) / f_range;
    
    % Find feasible center and best solution
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        feasible_center = mean(popdecs(feasible_mask, :), 1);
    else
        [~, min_cv_idx] = min(cons);
        feasible_center = popdecs(min_cv_idx, :);
    end
    
    % Identify best solution considering constraints
    [~, best_idx] = min(popfits + 1e6 * cv_pos);
    x_best = popdecs(best_idx, :);
    
    % Generate random indices for mutation
    rand_idx = zeros(NP, 6);
    for i = 1:NP
        available = setdiff(1:NP, [i, best_idx]);
        rand_idx(i,:) = available(randperm(length(available), 6));
    end
    
    % Calculate adaptive scaling factors
    F_feas = 0.5 * (1 - cv_norm);
    F_elite = 0.5 * (1 - f_norm);
    F_div = 0.3 * f_norm;
    
    % Calculate adaptive weights
    denom = 3 - cv_norm - f_norm + eps;
    w_feas = (1 - cv_norm) ./ denom;
    w_elite = (1 - f_norm) ./ denom;
    w_div = (1 + f_norm) ./ denom;
    
    % Vectorized mutation components
    diff_feas = bsxfun(@minus, feasible_center, popdecs);
    diff_best = bsxfun(@minus, x_best, popdecs);
    diff_elite1 = popdecs(rand_idx(:,1),:) - popdecs(rand_idx(:,2),:);
    diff_div1 = popdecs(rand_idx(:,3),:) - popdecs(rand_idx(:,4),:);
    diff_div2 = popdecs(rand_idx(:,5),:) - popdecs(rand_idx(:,6),:);
    
    % Combine components
    v_feas = popdecs + bsxfun(@times, F_feas, diff_feas);
    v_elite = popdecs + bsxfun(@times, F_elite, diff_best + diff_elite1);
    v_div = popdecs + bsxfun(@times, F_div, diff_div1 + diff_div2);
    
    mutants = bsxfun(@times, w_feas, v_feas) + ...
              bsxfun(@times, w_elite, v_elite) + ...
              bsxfun(@times, w_div, v_div);
    
    % Constraint repair for high CV solutions
    repair_mask = cv_norm > 0.7;
    if any(repair_mask)
        beta = 0.7 * cv_norm(repair_mask);
        perturbation = bsxfun(@times, (ub - lb), rand(sum(repair_mask),D) - 0.5);
        mutants(repair_mask,:) = bsxfun(@times, (1-beta), mutants(repair_mask,:)) + ...
                               bsxfun(@times, beta, bsxfun(@plus, feasible_center, 0.1*perturbation));
    end
    
    % Adaptive crossover
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
        
        % Final check to ensure within bounds
        offspring(below,j) = min(max(offspring(below,j), lb(j)), ub(j));
        offspring(above,j) = max(min(offspring(above,j), lb(j)), ub(j));
    end
    
    % Final feasibility enforcement for extreme cases
    out_of_bounds = any(bsxfun(@lt, offspring, lb) | bsxfun(@gt, offspring, ub), 2);
    if any(out_of_bounds)
        offspring(out_of_bounds,:) = bsxfun(@plus, feasible_center, ...
                                   0.1*bsxfun(@times, (ub - lb), rand(sum(out_of_bounds),D)-0.5));
    end
end