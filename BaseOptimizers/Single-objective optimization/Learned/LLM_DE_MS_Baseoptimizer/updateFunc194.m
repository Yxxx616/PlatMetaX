% MATLAB Code
function [offspring] = updateFunc194(popdecs, popfits, cons)
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
        F_fit = 0.8 * tanh((popfits(i) - avg_fit)/(max_fit - min_fit + eps));
        F_con = 0.4 * (1 - abs(cons(i))/sum_con);
        
        % Mutation components
        elite_diff = elite - popdecs(i, :);
        rand_diff = popdecs(r1(i), :) - popdecs(r2(i), :);
        best_worst_diff = x_best - x_worst;
        
        % Combined mutation
        offspring(i, :) = popdecs(i, :) + ...
            F_fit * elite_diff + ...
            F_con * rand_diff + ...
            0.1 * randn(1, D) .* best_worst_diff;
    end
    
    % Boundary handling with soft reflection
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    for i = 1:NP
        for j = 1:D
            if offspring(i,j) < lb(j)
                offspring(i,j) = 0.5*(popdecs(i,j) + lb(j));
            elseif offspring(i,j) > ub(j)
                offspring(i,j) = 0.5*(popdecs(i,j) + ub(j));
            end
        end
    end
    
    % Final clamping
    offspring = max(min(offspring, ub), lb);
end