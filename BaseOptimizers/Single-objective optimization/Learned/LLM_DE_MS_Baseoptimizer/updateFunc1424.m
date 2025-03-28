% MATLAB Code
function [offspring] = updateFunc1424(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Process constraints
    cv = max(0, cons); % Constraint violations
    feasible = cv <= 0;
    cv_max = max(cv) + eps;
    cv_norm = cv / cv_max;
    
    % Normalize fitness
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    f_norm = (popfits - f_min) / f_range;
    
    % Select elite individual (considering both fitness and constraints)
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    % Compute feasible center (weighted average)
    if any(feasible)
        weights = 1./(1 + cv(feasible));
        weights = weights / sum(weights);
        x_feas = sum(popdecs(feasible,:) .* weights(:), 1);
    else
        weights = 1./(1 + cv);
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
    
    % Adaptive scaling factors
    F_base = 0.5;
    F_adapt = F_base * (1 + 0.5*f_norm - 0.5*cv_norm);
    
    % Mutation components
    diff_elite = bsxfun(@minus, x_elite, popdecs);
    diff_feas = bsxfun(@minus, x_feas, popdecs);
    diff_div = popdecs(rand_idx1,:) - popdecs(rand_idx2,:);
    
    % Weighted mutation
    w_elite = 0.4 * (1 - f_norm);
    w_feas = 0.4 * (1 - cv_norm);
    w_div = 0.2 * (1 + cv_norm);
    
    mutants = popdecs + bsxfun(@times, F_adapt, ...
        w_elite.*diff_elite + w_feas.*diff_feas + w_div.*diff_div);
    
    % Boundary handling with reflection and reinitialization
    for j = 1:D
        % Reflection for slightly out-of-bound
        below = mutants(:,j) < lb(j);
        above = mutants(:,j) > ub(j);
        mutants(below,j) = 2*lb(j) - mutants(below,j);
        mutants(above,j) = 2*ub(j) - mutants(above,j);
        
        % Reinitialize for extreme violations
        extreme = (mutants(:,j) < lb(j)-10) | (mutants(:,j) > ub(j)+10);
        mutants(extreme,j) = lb(j) + (ub(j)-lb(j))*rand(sum(extreme),1);
    end
    
    % Adaptive crossover
    CR_base = 0.9;
    CR = CR_base - 0.4*cv_norm - 0.2*f_norm;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Enhanced exploration for highly constrained solutions
    high_cv = cv_norm > 0.8;
    if any(high_cv)
        n_high = sum(high_cv);
        offspring(high_cv,:) = bsxfun(@plus, x_feas, ...
            bsxfun(@times, (ub - lb), rand(n_high,D)-0.5) * 0.5);
    end
end