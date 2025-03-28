% MATLAB Code
function [offspring] = updateFunc1351(popdecs, popfits, cons)
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
    
    % Find feasible center
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        feasible_center = mean(popdecs(feasible_mask, :), 1);
    else
        [~, min_cv_idx] = min(cons);
        feasible_center = popdecs(min_cv_idx, :);
    end
    
    % Identify best solution (considering constraints)
    [~, best_idx] = min(popfits + 1e6 * cv_pos);
    x_best = popdecs(best_idx, :);
    
    % Generate unique random indices
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    invalid = (rand_idx1 == (1:NP)') | (rand_idx2 == (1:NP)') | ...
              (rand_idx1 == best_idx) | (rand_idx2 == best_idx);
    while any(invalid)
        rand_idx1(invalid) = randi(NP, sum(invalid), 1);
        rand_idx2(invalid) = randi(NP, sum(invalid), 1);
        invalid = (rand_idx1 == (1:NP)') | (rand_idx2 == (1:NP)') | ...
                  (rand_idx1 == best_idx) | (rand_idx2 == best_idx);
    end
    
    % Calculate adaptive weights
    w_b = 1 ./ (1 + exp(5 * cv_norm));  % Best component weight
    w_f = cv_norm ./ (1 + cv_norm);     % Feasible center weight
    w_r = 1 - w_b - w_f;                % Random component weight
    
    % Calculate adaptive scaling factors
    Fb = 0.7 * (1 - f_norm);
    Ff = 0.5 * cv_norm;
    Fr = 0.3 * (1 - f_norm) .* cv_norm;
    
    % Vectorized mutation components
    diff_best = x_best - popdecs;
    diff_center = feasible_center - popdecs;
    diff_rand = popdecs(rand_idx1,:) - popdecs(rand_idx2,:);
    
    % Combine components with weights
    v_b = popdecs + Fb.*diff_best;
    v_f = popdecs + Ff.*diff_center;
    v_r = popdecs + Fr.*diff_rand;
    
    mutants = v_b.*w_b + v_f.*w_f + v_r.*w_r;
    
    % Adaptive crossover
    CR = 0.9 * w_b + 0.1;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Constraint-guided repair
    repair_mask = cv_pos > 0;
    if any(repair_mask)
        beta = 0.5 * cv_norm(repair_mask);
        offspring(repair_mask,:) = (1-beta).*offspring(repair_mask,:) + ...
                                 beta.*feasible_center;
    end
    
    % Boundary handling with reflection
    for j = 1:D
        below = offspring(:,j) < lb(j);
        above = offspring(:,j) > ub(j);
        offspring(below,j) = 2*lb(j) - offspring(below,j);
        offspring(above,j) = 2*ub(j) - offspring(above,j);
        
        % Final check to ensure within bounds
        offspring(below,j) = min(max(offspring(below,j), lb(j));
        offspring(above,j) = max(min(offspring(above,j), ub(j));
    end
    
    % Final feasibility enforcement
    out_of_bounds = any(offspring < lb(ones(NP,1),:) | offspring > ub(ones(NP,1),:), 2);
    if any(out_of_bounds)
        offspring(out_of_bounds,:) = lb + rand(sum(out_of_bounds),D).*(ub-lb);
    end
end