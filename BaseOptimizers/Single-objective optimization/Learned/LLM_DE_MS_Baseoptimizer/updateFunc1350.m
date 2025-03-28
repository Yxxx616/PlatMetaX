% MATLAB Code
function [offspring] = updateFunc1350(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    eps = 1e-12;
    
    % Normalize constraints
    cv_pos = max(0, cons);
    cv_max = max(cv_pos) + eps;
    cv_norm = cv_pos / cv_max;
    
    % Normalize fitness (minimization)
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    f_norm = (popfits - f_min) / f_range;
    
    % Feasibility sigmoid weighting
    w_f = 1 ./ (1 + exp(5 * cv_norm));
    
    % Find feasible center
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        feasible_center = mean(popdecs(feasible_mask, :), 1);
    else
        [~, min_cv_idx] = min(cons);
        feasible_center = popdecs(min_cv_idx, :);
    end
    
    % Identify best solution
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
    
    % Calculate adaptive factors
    Fb = 0.5 * w_f .* (1 - f_norm);
    Ff = 0.3 * (1 - w_f);
    Fr = 0.2 * (1 - f_norm) .* (1 - w_f);
    
    % Vectorized mutation
    diff_best = x_best - popdecs;
    diff_center = feasible_center - popdecs;
    diff_rand = popdecs(rand_idx1,:) - popdecs(rand_idx2,:);
    
    mutants = popdecs + Fb.*diff_best + Ff.*diff_center + Fr.*diff_rand;
    
    % Adaptive crossover
    CR = 0.9 * w_f + 0.1;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Constraint-guided repair
    repair_mask = cv_pos > 0;
    if any(repair_mask)
        beta = 0.7 * cv_norm(repair_mask);
        offspring(repair_mask,:) = (1-beta).*offspring(repair_mask,:) + ...
                                 beta.*feasible_center;
    end
    
    % Boundary handling with reflection
    for j = 1:D
        below = offspring(:,j) < lb(j);
        above = offspring(:,j) > ub(j);
        offspring(below,j) = lb(j) + rand(sum(below),1).*(ub(j)-lb(j));
        offspring(above,j) = ub(j) - rand(sum(above),1).*(ub(j)-lb(j));
    end
    
    % Final feasibility enforcement
    out_of_bounds = any(offspring < lb(ones(NP,1),:) | offspring > ub(ones(NP,1),:), 2);
    if any(out_of_bounds)
        offspring(out_of_bounds,:) = lb + rand(sum(out_of_bounds),D).*(ub-lb);
    end
end