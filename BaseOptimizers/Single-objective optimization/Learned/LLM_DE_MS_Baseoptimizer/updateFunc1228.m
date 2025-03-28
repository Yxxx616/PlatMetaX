% MATLAB Code
function [offspring] = updateFunc1228(popdecs, popfits, cons)
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
    
    % 3. Generate random indices for differential vectors
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    invalid = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    while any(invalid)
        r1(invalid) = randi(NP, sum(invalid), 1);
        r2(invalid) = randi(NP, sum(invalid), 1);
        invalid = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    end
    
    % 4. Calculate direction vectors with perturbation
    dist_to_best = sqrt(sum((popdecs - x_best(ones(NP,1), :)).^2, 2));
    sigma = 0.1 * dist_to_best;
    noise = randn(NP, D) .* sigma(:, ones(1, D));
    
    best_dir = x_best(ones(NP,1), :) - popdecs;
    diff_dir = popdecs(r1,:) - popdecs(r2,:);
    directions = 0.7*best_dir + 0.3*diff_dir + 0.2*noise;
    
    % 5. Adaptive scaling factors
    F = 0.5 + 0.3*(cv_abs./(max_cv + eps_val)) + ...
        0.2*((popfits - f_min)./(f_max - f_min + eps_val));
    F = F(:, ones(1, D));
    
    % 6. Mutation
    mutants = popdecs + F .* directions;
    
    % 7. Adaptive crossover
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