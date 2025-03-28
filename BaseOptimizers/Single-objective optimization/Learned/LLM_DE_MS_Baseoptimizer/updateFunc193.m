% MATLAB Code
function [offspring] = updateFunc193(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Identify elite (best feasible or least infeasible)
    [~, elite_idx] = min(popfits + 1e6*abs(cons));
    elite = popdecs(elite_idx, :);
    
    % Identify best and worst individuals (constrained)
    [~, sorted_idx] = sort(popfits + 1e6*abs(cons));
    x_best = popdecs(sorted_idx(1), :);
    x_worst = popdecs(sorted_idx(end), :);
    
    % Normalize constraints and fitness
    max_con = max(abs(cons)) + eps;
    sum_con = sum(abs(cons)) + eps;
    min_fit = min(popfits);
    max_fit = max(popfits);
    avg_fit = mean(popfits);
    
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
        F_elite = 0.9 * (1 - abs(cons(i))/max_con);
        F_fit = 0.5 * tanh((popfits(i) - avg_fit)/(max_fit - min_fit + eps));
        F_con = 0.3 * (1 - abs(cons(i))/sum_con);
        
        elite_diff = elite - popdecs(i, :);
        fit_diff = x_best - x_worst;
        rand_diff = popdecs(r1(i), :) - popdecs(r2(i), :);
        
        offspring(i, :) = popdecs(i, :) + ...
            F_elite * elite_diff + ...
            F_fit * fit_diff + ...
            F_con * rand_diff;
    end
    
    % Boundary handling with reflection
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Vectorized boundary control
    mask_low = offspring < lb;
    mask_high = offspring > ub;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
        (2*lb - offspring) .* mask_low + ...
        (2*ub - offspring) .* mask_high;
    
    % Final clamping
    offspring = max(min(offspring, ub), lb);
end