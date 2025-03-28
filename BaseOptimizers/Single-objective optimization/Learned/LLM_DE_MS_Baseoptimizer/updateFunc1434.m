% MATLAB Code
function [offspring] = updateFunc1434(popdecs, popfits, cons)
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
    
    % Enhanced adaptive scaling factors
    F_elite = 0.8 * (1 - 0.5*f_norm - 0.5*cv_norm);
    F_feas = 0.6 * (1 - 0.3*f_norm + 0.7*cv_norm);
    F_div = 0.4 * (1 + 0.1*f_norm - 0.2*cv_norm);
    F_rand = 0.2 * (0.5 + 0.5*cv_norm);
    
    % Temperature-controlled weights
    T = [0.1, 0.2, 0.3, 0.4]; % Temperature parameters
    w = exp(-(f_norm + cv_norm)./T);
    w = w ./ sum(w, 2);
    
    % Mutation components
    diff_elite = x_elite - popdecs;
    diff_feas = x_feas - popdecs;
    diff_div = popdecs(rand_idx1,:) - popdecs(rand_idx2,:);
    rand_pert = (ub - lb) .* (rand(NP,D)-0.5);
    
    % Combined mutation with dynamic weighting
    mutants = popdecs + ...
        w(:,1).*F_elite.*diff_elite + ...
        w(:,2).*F_feas.*diff_feas + ...
        w(:,3).*F_div.*diff_div + ...
        w(:,4).*F_rand.*rand_pert;
    
    % Improved boundary handling with reflection
    for j = 1:D
        below = mutants(:,j) < lb(j);
        above = mutants(:,j) > ub(j);
        mutants(below,j) = 2*lb(j) - mutants(below,j);
        mutants(above,j) = 2*ub(j) - mutants(above,j);
        
        % Reinitialize if still out of bounds
        still_bad = (mutants(:,j) < lb(j)) | (mutants(:,j) > ub(j));
        mutants(still_bad,j) = lb(j) + (ub(j)-lb(j)).*rand(sum(still_bad),1);
    end
    
    % Dynamic crossover rate with jitter
    CR_base = 0.85;
    CR = CR_base - 0.4*cv_norm - 0.3*f_norm;
    CR = min(max(CR, 0.1), 0.95);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Special handling for highly constrained solutions
    high_cv = cv_norm > 0.9;
    if any(high_cv)
        n_high = sum(high_cv);
        offspring(high_cv,:) = x_feas + (ub - lb).*(rand(n_high,D)-0.5)*0.1;
    end
end