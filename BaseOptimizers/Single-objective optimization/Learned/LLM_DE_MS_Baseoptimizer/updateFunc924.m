% MATLAB Code
function [offspring] = updateFunc924(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Constraint-weighted fitness calculation
    f_range = max(popfits) - min(popfits);
    weighted_fits = popfits + 0.5 * max(0, cons) * f_range;
    
    % 2. Select best individual considering constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(weighted_fits(feasible));
        x_best = popdecs(feasible(best_idx),:);
    else
        [~, best_idx] = min(weighted_fits);
        x_best = popdecs(best_idx,:);
    end
    
    % 3. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % 4. Adaptive parameters
    cons_norm = (cons - min(cons)) ./ (max(cons) - min(cons) + eps);
    [~, rank_idx] = sort(weighted_fits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    
    F = 0.3 + 0.5 * (ranks/NP);
    CR = 0.7 + 0.2 * cons_norm;
    
    % 5. Enhanced mutation with constraint adaptation
    beta = 0.5 * (1 - abs(cons_norm));
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    x_best_rep = repmat(x_best, NP, 1);
    
    mutants = x_best_rep + F.*diff1 + beta.*diff2;
    
    % 6. Hybrid crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % 7. Create offspring with improved boundary handling
    offspring = popdecs;
    gamma = 0.3;
    update_mask = mask;
    offspring(update_mask) = mutants(update_mask);
    
    % Partial update for non-crossover dimensions
    partial_update = ~mask & (rand(NP, D) < 0.1;
    offspring(partial_update) = popdecs(partial_update) + ...
        gamma .* (mutants(partial_update) - popdecs(partial_update));
    
    % 8. Adaptive boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection with adaptive scaling
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    scale = 0.2 + 0.6 * rand(NP, D);
    offspring(below) = lb_rep(below) + scale(below) .* (popdecs(below) - lb_rep(below));
    offspring(above) = ub_rep(above) - scale(above) .* (ub_rep(above) - popdecs(above));
    
    % Final clamping
    offspring = max(min(offspring, ub_rep), lb_rep);
end