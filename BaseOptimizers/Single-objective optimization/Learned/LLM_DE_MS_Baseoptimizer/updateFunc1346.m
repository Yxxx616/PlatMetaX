% MATLAB Code
function [offspring] = updateFunc1346(popdecs, popfits, cons)
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
    
    % Identify feasible solutions and center
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        feasible_center = mean(popdecs(feasible_mask, :), 1);
    else
        [~, min_cv_idx] = min(cons);
        feasible_center = popdecs(min_cv_idx, :);
    end
    
    % Find best solution (considering constraints)
    [~, best_idx] = min(popfits + 1e6 * cv_pos);
    x_best = popdecs(best_idx, :);
    
    % Pre-generate random indices for vectorization
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    invalid = (rand_idx1 == (1:NP)') | (rand_idx2 == (1:NP)');
    while any(invalid)
        rand_idx1(invalid) = randi(NP, sum(invalid), 1);
        rand_idx2(invalid) = randi(NP, sum(invalid), 1);
        invalid = (rand_idx1 == (1:NP)') | (rand_idx2 == (1:NP)');
    end
    
    % Calculate mutation factors
    F1 = 0.7 * (1 - f_norm) .* (1 - cv_norm);
    F2 = 0.5 * cv_norm;
    F3 = 0.3 * f_norm .* cv_norm;
    
    % Vectorized mutation
    diff_best = x_best - popdecs;
    diff_center = feasible_center - popdecs;
    diff_rand = popdecs(rand_idx1,:) - popdecs(rand_idx2,:);
    
    mutants = popdecs + F1.*diff_best + F2.*diff_center + F3.*diff_rand;
    
    % Adaptive crossover rates
    CR = 0.9 * (1 - cv_norm) + 0.1;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Constraint repair for infeasible solutions
    repair_mask = cv_pos > 0;
    if any(repair_mask)
        beta = 0.4 * cv_norm(repair_mask);
        offspring(repair_mask,:) = (1-beta).*offspring(repair_mask,:) + ...
                                  beta.*feasible_center;
    end
    
    % Boundary handling with reflection and randomization
    for j = 1:D
        % Reflection
        below = offspring(:,j) < lb(j);
        above = offspring(:,j) > ub(j);
        offspring(below,j) = 2*lb(j) - offspring(below,j);
        offspring(above,j) = 2*ub(j) - offspring(above,j);
        
        % Randomization if still out of bounds
        still_out = (offspring(:,j) < lb(j)) | (offspring(:,j) > ub(j));
        if any(still_out)
            offspring(still_out,j) = lb(j) + rand(sum(still_out),1).*(ub(j)-lb(j));
        end
    end
end