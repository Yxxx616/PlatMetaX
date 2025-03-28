% MATLAB Code
function [offspring] = updateFunc186(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Identify feasible solutions and elite
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
    r1 = zeros(NP, 1);
    r2 = zeros(NP, 1);
    for i = 1:NP
        available = setdiff(1:NP, i);
        selected = available(randperm(length(available), 2));
        r1(i) = selected(1);
        r2(i) = selected(2);
    end
    
    % Vectorized mutation
    for i = 1:NP
        F1 = 0.9 * (1 - norm_cons(i));
        F2 = 0.5 * norm_fits(i);
        F3 = 0.3 * norm_cons(i);
        
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