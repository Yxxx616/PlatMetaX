% MATLAB Code
function [offspring] = updateFunc195(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Identify elite individual (best feasible or least infeasible)
    [~, elite_idx] = min(popfits + 1e6*abs(cons));
    elite = popdecs(elite_idx, :);
    
    % Identify best and worst individuals (constrained)
    [~, sorted_idx] = sort(popfits + 1e6*abs(cons));
    x_best = popdecs(sorted_idx(1), :);
    x_worst = popdecs(sorted_idx(end), :);
    
    % Calculate statistics
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
        % Adaptive scaling factors
        F1 = 0.5 + 0.3 * tanh((popfits(i) - avg_fit)/(max_fit - min_fit + eps));
        F2 = 0.3 + 0.2 * (1 - abs(cons(i))/sum_con);
        F3 = 0.2 * (1 - (popfits(i) - min_fit)/(max_fit - min_fit + eps));
        
        % Mutation components
        elite_diff = elite - popdecs(i, :);
        rand_diff = popdecs(r1(i), :) - popdecs(r2(i), :);
        best_worst_diff = x_best - x_worst;
        
        % Combined mutation
        offspring(i, :) = popdecs(i, :) + ...
            F1 * elite_diff + ...
            F2 * rand_diff + ...
            F3 * randn(1, D) .* best_worst_diff;
    end
    
    % Boundary handling with adaptive reflection
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    for i = 1:NP
        for j = 1:D
            if offspring(i,j) < lb(j)
                offspring(i,j) = lb(j) + 0.5*rand*(popdecs(i,j) - lb(j));
            elseif offspring(i,j) > ub(j)
                offspring(i,j) = ub(j) - 0.5*rand*(ub(j) - popdecs(i,j));
            end
        end
    end
    
    % Final clamping with small random perturbation
    offspring = max(min(offspring, ub), lb) + 0.01*randn(size(offspring)).*(ub-lb);
end