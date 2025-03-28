% MATLAB Code
function [offspring] = updateFunc1677(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    abs_cons = abs(cons);
    max_cons = max(abs_cons);
    min_fit = min(popfits);
    max_fit = max(popfits);
    
    norm_cons = abs_cons ./ (max_cons + eps);
    norm_fits = (popfits - min_fit) ./ (max_fit - min_fit + eps);
    
    % Identify feasible solutions
    feasible_mask = cons <= 0;
    feasible_pop = popdecs(feasible_mask, :);
    feasible_fits = popfits(feasible_mask);
    
    % Find best solution (prioritize feasible)
    if any(feasible_mask)
        [~, best_idx] = min(feasible_fits);
        x_best = feasible_pop(best_idx, :);
    else
        [~, best_idx] = min(popfits + norm_cons*100);
        x_best = popdecs(best_idx, :);
    end
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    % Pre-compute random indices
    [~, rand_idx] = sort(rand(NP, NP), 2);
    rand_idx = rand_idx(:, 2:3);
    
    % Compute scaling factors
    F = zeros(NP, 1);
    F(feasible_mask) = 0.5 + 0.3 * norm_fits(feasible_mask);
    F(~feasible_mask) = 0.3 + 0.5 * norm_cons(~feasible_mask);
    
    % Compute perturbation strength
    sigma = 0.2 * (1 + norm_cons);
    
    for i = 1:NP
        % Select random individuals
        r1 = rand_idx(i,1); 
        r2 = rand_idx(i,2);
        
        % Mutation
        mutation = popdecs(i,:) + ...
                  F(i) * (x_best - popdecs(i,:)) + ...
                  (1-F(i)) * (popdecs(r1,:) - popdecs(r2,:)) + ...
                  sigma(i) * randn(1, D);
        
        % Adaptive crossover
        CR = 0.9 * (1 - norm_cons(i)) + 0.1 * norm_fits(i);
        j_rand = randi(D);
        mask = rand(1, D) < CR;
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
    end
    
    % Boundary handling with midpoint reflection
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    
    offspring(viol_low) = (lb(viol_low) + popdecs(viol_low)) / 2;
    offspring(viol_high) = (ub(viol_high) + popdecs(viol_high)) / 2;
    
    % Final bounds check
    offspring = max(min(offspring, ub), lb);
end