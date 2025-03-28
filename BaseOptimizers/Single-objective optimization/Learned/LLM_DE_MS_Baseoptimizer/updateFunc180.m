function [offspring] = updateFunc180(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Identify feasible solutions
    feasible_mask = cons <= 0;
    feasible_pop = popdecs(feasible_mask, :);
    feasible_fits = popfits(feasible_mask);
    
    % Normalize fitness and constraints
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    norm_cons = abs(cons) ./ (max(abs(cons)) + eps);
    
    % Determine elite solution (best feasible or overall best)
    if ~isempty(feasible_pop)
        [~, elite_idx] = min(feasible_fits); % Minimization problem
        elite = feasible_pop(elite_idx, :);
    else
        [~, elite_idx] = min(popfits);
        elite = popdecs(elite_idx, :);
    end
    
    % Determine current best and worst solutions
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    % Pre-compute random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    same_idx = (r1 == r2);
    r2(same_idx) = mod(r2(same_idx) + randi(NP-1), NP) + 1;
    
    % Vectorized mutation
    for i = 1:NP
        % Adaptive scaling factors
        F1 = 0.8 * (1 - norm_cons(i)); % Decreases with constraint violation
        F2 = 0.6 * (1 - norm_fits(i)); % Increases with better fitness
        F3 = 0.4 * norm_cons(i); % Increases with constraint violation
        
        % Mutation components
        elite_diff = elite - popdecs(i, :);
        fit_diff = x_best - x_worst;
        rand_diff = popdecs(r1(i), :) - popdecs(r2(i), :);
        
        % Combined mutation
        offspring(i, :) = popdecs(i, :) + F1*elite_diff + F2*fit_diff + F3*rand_diff;
    end
    
    % Boundary control
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = min(max(offspring, lb), ub);
end