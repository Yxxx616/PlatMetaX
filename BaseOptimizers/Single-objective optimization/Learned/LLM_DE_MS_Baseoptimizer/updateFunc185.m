function [offspring] = updateFunc185(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Identify feasible solutions
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        feasible_pop = popdecs(feasible_mask, :);
        feasible_fits = popfits(feasible_mask);
        [~, elite_idx] = min(feasible_fits);
        elite = feasible_pop(elite_idx, :);
    else
        [~, elite_idx] = min(popfits);
        elite = popdecs(elite_idx, :);
    end
    
    % Normalize fitness and constraints
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    norm_cons = abs(cons) ./ (max(abs(cons)) + eps);
    
    % Determine current best and average
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    x_avg = mean(popdecs, 1);
    
    % Generate random indices (r1 ≠ r2 ≠ i)
    r1 = arrayfun(@(i) setdiff(randperm(NP, 2), i), 1:NP, 'UniformOutput', false);
    r2 = cellfun(@(x) x(1), r1);
    r1 = cellfun(@(x) x(2), r1);
    
    % Vectorized mutation
    for i = 1:NP
        F1 = 0.8 * (1 - norm_cons(i));
        F2 = 0.6 * norm_fits(i);
        F3 = 0.4 * norm_cons(i);
        
        elite_diff = elite - popdecs(i, :);
        fit_diff = x_best - x_avg;
        rand_diff = popdecs(r1(i), :) - popdecs(r2(i), :);
        
        offspring(i, :) = popdecs(i, :) + F1*elite_diff + F2*fit_diff + F3*rand_diff;
    end
    
    % Boundary control
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = min(max(offspring, lb), ub);
end