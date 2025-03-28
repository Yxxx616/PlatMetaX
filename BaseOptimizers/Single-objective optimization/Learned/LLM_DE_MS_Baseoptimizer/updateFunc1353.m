% MATLAB Code
function [offspring] = updateFunc1353(popdecs, popfits, cons)
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
    
    % Generate random indices for diversity component
    rand_idx = zeros(NP, 4);
    for i = 1:NP
        available = setdiff(1:NP, [i, best_idx]);
        rand_idx(i,:) = available(randperm(length(available), 4));
    end
    
    % Calculate adaptive weights
    w_b = 1 ./ (1 + cv_norm);          % Best component weight
    w_c = cv_norm ./ (1 + cv_norm);    % Center component weight
    w_d = 1 - w_b - w_c;               % Diversity component weight
    
    % Calculate adaptive scaling factors
    F_b = 0.8 * (1 - f_norm);
    F_c = 0.6 * cv_norm;
    F_d = 0.4 * (1 - f_norm) .* cv_norm;
    
    % Vectorized mutation components
    diff_best = x_best - popdecs;
    diff_center = feasible_center - popdecs;
    diff_div1 = popdecs(rand_idx(:,1),:) - popdecs(rand_idx(:,2),:);
    diff_div2 = popdecs(rand_idx(:,3),:) - popdecs(rand_idx(:,4),:);
    
    % Combine components with weights
    v_b = popdecs + F_b.*diff_best;
    v_c = popdecs + F_c.*diff_center;
    v_d = popdecs + F_d.*(diff_div1 + diff_div2);
    
    mutants = v_b.*w_b + v_c.*w_c + v_d.*w_d;
    
    % Adaptive crossover
    CR = 0.9 * w_b + 0.1;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Enhanced constraint-guided repair
    repair_mask = cv_pos > 0;
    if any(repair_mask)
        beta = 0.5 * cv_norm(repair_mask);
        offspring(repair_mask,:) = (1-beta).*offspring(repair_mask,:) + ...
                                 beta.*(feasible_center + 0.1*randn(sum(repair_mask),D));
    end
    
    % Boundary handling with adaptive reflection
    for j = 1:D
        below = offspring(:,j) < lb(j);
        above = offspring(:,j) > ub(j);
        
        % Adaptive reflection based on constraint violation
        reflect_factor = 1 + 0.5*cv_norm;
        offspring(below,j) = (2 - reflect_factor(below)).*lb(j) - offspring(below,j);
        offspring(above,j) = (2 - reflect_factor(above)).*ub(j) - offspring(above,j);
        
        % Final check to ensure within bounds
        offspring(below,j) = min(max(offspring(below,j), lb(j)), ub(j));
        offspring(above,j) = max(min(offspring(above,j), lb(j)), ub(j));
    end
    
    % Final feasibility enforcement with random reset
    out_of_bounds = any(offspring < lb(ones(NP,1),:) | offspring > ub(ones(NP,1),:), 2);
    if any(out_of_bounds)
        offspring(out_of_bounds,:) = repmat(lb, sum(out_of_bounds), 1) + ...
                                   rand(sum(out_of_bounds),D).*repmat(ub-lb, sum(out_of_bounds), 1);
    end
end