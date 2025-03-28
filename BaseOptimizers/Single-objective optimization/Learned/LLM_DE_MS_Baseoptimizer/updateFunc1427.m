% MATLAB Code
function [offspring] = updateFunc1427(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Process constraints
    cv = max(0, cons); % Constraint violations
    feasible = cv <= 0;
    cv_max = max(cv) + eps;
    cv_norm = cv / cv_max;
    
    % Normalize fitness (inverted for minimization)
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    f_norm = (popfits - f_min) / f_range;
    
    % Select elite individual (considering both fitness and constraints)
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    % Compute feasible center (weighted average)
    if any(feasible)
        weights = exp(-cv(feasible));
        weights = weights / sum(weights);
        x_feas = sum(popdecs(feasible,:) .* weights(:), 1);
    else
        weights = exp(-cv);
        weights = weights / sum(weights);
        x_feas = sum(popdecs .* weights(:), 1);
    end
    
    % Generate random indices for diversity component
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    invalid = (rand_idx1 == rand_idx2) | (rand_idx1 == (1:NP)') | (rand_idx2 == (1:NP)');
    while any(invalid)
        rand_idx1(invalid) = randi(NP, sum(invalid), 1);
        rand_idx2(invalid) = randi(NP, sum(invalid), 1);
        invalid = (rand_idx1 == rand_idx2) | (rand_idx1 == (1:NP)') | (rand_idx2 == (1:NP)');
    end
    
    % Improved adaptive scaling factors
    F1 = 0.7 * (1 - 0.5*f_norm - 0.3*cv_norm);
    F2 = 0.5 * (1 - 0.4*f_norm + 0.4*cv_norm);
    F3 = 0.3 * (1 + 0.3*f_norm - 0.2*cv_norm);
    
    % Mutation components with dynamic weights
    w1 = 1./(1 + f_norm + cv_norm);
    w2 = 1./(1 + f_norm - 0.5*cv_norm);
    w3 = 1./(2 - f_norm + 0.5*cv_norm);
    w_sum = w1 + w2 + w3;
    w1 = w1 ./ w_sum;
    w2 = w2 ./ w_sum;
    w3 = w3 ./ w_sum;
    
    diff_elite = x_elite - popdecs;
    diff_feas = x_feas - popdecs;
    diff_div = popdecs(rand_idx1,:) - popdecs(rand_idx2,:);
    
    % Combined mutation with dynamic weighting
    mutants = popdecs + w1.*(F1.*diff_elite) + w2.*(F2.*diff_feas) + w3.*(F3.*diff_div);
    
    % Boundary handling with adaptive reflection
    for j = 1:D
        below = mutants(:,j) < lb(j);
        above = mutants(:,j) > ub(j);
        mutants(below,j) = lb(j) + (ub(j)-lb(j))*rand(sum(below),1);
        mutants(above,j) = lb(j) + (ub(j)-lb(j))*rand(sum(above),1);
    end
    
    % Adaptive crossover with dynamic CR
    CR_base = 0.85;
    CR = CR_base - 0.5*cv_norm - 0.3*f_norm;
    CR = min(max(CR, 0.1), 0.95); % Clamp CR values
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Enhanced exploration for highly constrained solutions
    high_cv = cv_norm > 0.9;
    if any(high_cv)
        n_high = sum(high_cv);
        offspring(high_cv,:) = x_feas + (ub - lb).*(rand(n_high,D)-0.5)*0.3;
    end
end