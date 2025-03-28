% MATLAB Code
function [offspring] = updateFunc187(popdecs, popfits, cons)
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
        [~, elite_idx] = min(popfits + 1e6*abs(cons)); % penalty method
        elite = popdecs(elite_idx, :);
    end
    
    % Normalize fitness and constraints
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    norm_cons = abs(cons) ./ (max(abs(cons)) + eps);
    
    % Determine current best and average
    [~, best_idx] = min(popfits + 1e6*abs(cons)); % constrained best
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
    
    % Vectorized mutation with adaptive weights
    for i = 1:NP
        F1 = 0.8 * (1 - norm_cons(i)); % Elite guidance weight
        F2 = 0.6 * norm_fits(i);      % Fitness-directed weight
        F3 = 0.4 * norm_cons(i);       % Constraint exploration weight
        
        elite_diff = elite - popdecs(i, :);
        fit_diff = x_best - x_avg;
        rand_diff = popdecs(r1(i), :) - popdecs(r2(i), :);
        
        offspring(i, :) = popdecs(i, :) + F1*elite_diff + F2*fit_diff + F3*rand_diff;
    end
    
    % Boundary control with reflection
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    for i = 1:NP
        for j = 1:D
            if offspring(i,j) < lb(j)
                offspring(i,j) = lb(j) + (lb(j) - offspring(i,j));
                if offspring(i,j) > ub(j)
                    offspring(i,j) = ub(j);
                end
            elseif offspring(i,j) > ub(j)
                offspring(i,j) = ub(j) - (offspring(i,j) - ub(j));
                if offspring(i,j) < lb(j)
                    offspring(i,j) = lb(j);
                end
            end
        end
    end
end