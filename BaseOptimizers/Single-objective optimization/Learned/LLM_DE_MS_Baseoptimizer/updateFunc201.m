% MATLAB Code
function [offspring] = updateFunc201(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Identify elite individual (best feasible or least infeasible)
    [~, elite_idx] = min(popfits + 1e6*abs(cons));
    elite = popdecs(elite_idx, :);
    
    % Sort population by constrained fitness
    [~, sorted_idx] = sort(popfits + 1e6*abs(cons));
    x_best = popdecs(sorted_idx(1), :);
    x_worst = popdecs(sorted_idx(end), :);
    
    % Calculate statistics
    sum_con = sum(abs(cons)) + eps;
    max_con = max(abs(cons)) + eps;
    min_fit = min(popfits);
    max_fit = max(popfits);
    avg_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    
    % Generate random indices (r1 ≠ r2 ≠ i)
    r1 = randi(NP-1, NP, 1);
    r2 = randi(NP-2, NP, 1);
    for i = 1:NP
        r1(i) = r1(i) + (r1(i) >= i);
        temp = randi(NP-2, 1);
        r2(i) = temp + (temp >= min(i, r1(i))) + (temp >= max(i, r1(i))-1);
    end
    
    % Vectorized mutation with adaptive weights
    for i = 1:NP
        % Adaptive scaling factors
        F1 = 0.8 * (1 - abs(cons(i))/max_con);
        F2 = 0.5 * (1 - (popfits(i) - min_fit)/(max_fit - min_fit + eps));
        F3 = 0.2 * tanh((popfits(i) - avg_fit)/std_fit);
        
        % Mutation components
        elite_diff = elite - popdecs(i, :);
        rand_diff = popdecs(r1(i), :) - popdecs(r2(i), :);
        best_worst_diff = x_best - x_worst;
        
        % Combined mutation
        offspring(i, :) = popdecs(i, :) + ...
            F1 .* elite_diff + ...
            F2 .* rand_diff .* (1 + abs(cons(i))/sum_con) + ...
            F3 .* randn(1, D) .* best_worst_diff .* ((popfits(i) - avg_fit)/std_fit;
    end
    
    % Boundary handling with adaptive reflection
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    out_of_bounds = (offspring < lb) | (offspring > ub);
    if any(out_of_bounds(:))
        beta = 0.1 + 0.8*rand(NP, D);
        reflection_low = lb + beta .* (popdecs - lb);
        reflection_high = ub - beta .* (ub - popdecs);
        offspring = offspring .* ~out_of_bounds + ...
                   reflection_low .* (offspring < lb) + ...
                   reflection_high .* (offspring > ub);
    end
    
    % Final adaptive perturbation
    perturbation_scale = 0.01*(max_fit - min_fit)/(std_fit + eps);
    perturbation = perturbation_scale * randn(NP, D);
    offspring = max(min(offspring + perturbation, ub), lb);
end