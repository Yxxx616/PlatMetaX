% MATLAB Code
function [offspring] = updateFunc191(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Identify elite individual (best feasible or least infeasible)
    [~, elite_idx] = min(popfits + 1e6*abs(cons));
    elite = popdecs(elite_idx, :);
    
    % Identify current best (constrained)
    [~, best_idx] = min(popfits + 1e6*abs(cons));
    x_best = popdecs(best_idx, :);
    x_avg = mean(popdecs, 1);
    
    % Normalized constraints
    max_cons = max(abs(cons)) + eps;
    norm_cons = abs(cons) / max_cons;
    
    % Sum of constraints and fitness for weights
    sum_cons = sum(abs(cons)) + eps;
    sum_fits = sum(abs(popfits)) + eps;
    
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
        F1 = 0.7 * (1 - norm_cons(i));  % Elite guidance weight
        F2 = 0.5 + 0.3 * tanh(cons(i)/sum_cons);  % Constraint exploration
        F3 = 0.4 * (1 - popfits(i)/sum_fits);  % Fitness-directed
        
        elite_diff = elite - popdecs(i, :);
        rand_diff = popdecs(r1(i), :) - popdecs(r2(i), :);
        fit_diff = x_best - x_avg;
        
        offspring(i, :) = popdecs(i, :) + F1*elite_diff + F2*rand_diff + F3*fit_diff;
    end
    
    % Boundary control with reflection
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Vectorized boundary handling
    mask_low = offspring < lb;
    mask_high = offspring > ub;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
                (2*lb - offspring) .* mask_low + ...
                (2*ub - offspring) .* mask_high;
    
    % Final clamping to ensure within bounds
    offspring = max(min(offspring, ub), lb);
end