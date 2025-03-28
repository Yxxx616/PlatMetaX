% MATLAB Code
function [offspring] = updateFunc1227(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps_val = 1e-10;
    
    % 1. Find best solution considering constraints
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        best_idx = temp(best_idx);
    else
        [~, best_idx] = min(cons);
    end
    x_best = popdecs(best_idx, :);
    
    % 2. Calculate constraint violation weights
    cv_abs = abs(cons);
    max_cv = max(cv_abs);
    f_min = min(popfits);
    f_max = max(popfits);
    
    % 3. Adaptive scaling factors
    F = 0.5 + 0.5 * (cv_abs./(max_cv + eps_val)) .* ...
           (1 - (popfits - f_min)./(f_max - f_min + eps_val));
    F = F(:, ones(1, D));
    
    % 4. Direction vectors with perturbation
    dist_to_best = sqrt(sum((popdecs - x_best(ones(NP,1), :)).^2, 2));
    sigma = 0.2 * dist_to_best;
    noise = randn(NP, D) .* sigma(:, ones(1, D));
    directions = x_best(ones(NP,1), :) - popdecs + noise;
    
    % 5. Mutation
    mutants = popdecs + F .* directions;
    
    % 6. Adaptive crossover
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.9 * (1 - ranks/NP) .* (1 - cv_abs/(max_cv + eps_val));
    CR = CR(:, ones(1, D));
    
    % Perform crossover
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end