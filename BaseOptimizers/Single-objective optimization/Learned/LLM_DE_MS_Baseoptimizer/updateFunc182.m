function [offspring] = updateFunc182(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Normalize fitness and constraints
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    norm_cons = abs(cons) ./ (max(abs(cons)) + eps);
    
    % Identify feasible solutions
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx), :);
    else
        [~, elite_idx] = min(popfits);
        elite = popdecs(elite_idx, :);
    end
    
    % Determine current best and worst solutions
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    same_idx = (r1 == (1:NP)');
    r2(same_idx) = mod(r2(same_idx) + randi(NP-1), NP) + 1;
    
    % Vectorized mutation
    for i = 1:NP
        % Adaptive scaling factors
        F1 = 0.8 * (1 - norm_cons(i));
        F2 = 0.6 * norm_fits(i);
        F3 = 0.4 * norm_cons(i);
        F4 = 0.2 * (1 - norm_fits(i));
        
        % Mutation components
        elite_diff = elite - popdecs(i, :);
        fit_diff = x_best - x_worst;
        rand_diff = popdecs(r1(i), :) - popdecs(r2(i), :);
        random_perturb = randn(1, D);
        
        % Combined mutation
        offspring(i, :) = popdecs(i, :) + F1*elite_diff + F2*fit_diff + F3*rand_diff + F4*random_perturb;
    end
    
    % Boundary control
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = min(max(offspring, lb), ub);
end