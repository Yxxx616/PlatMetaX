% MATLAB Code
function [offspring] = updateFunc566(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints
    norm_cons = abs(cons) ./ (max(abs(cons)) + eps);
    
    % Select elite considering constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, min_cons_idx] = min(cons);
        elite = popdecs(min_cons_idx, :);
    end
    
    % Rank population based on fitness
    [~, rank_order] = sort(popfits);
    [~, ranks] = sort(rank_order);
    norm_ranks = ranks / NP;
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    
    % Adaptive scaling factors
    F1 = 0.7 * (1 - norm_ranks) .* (1 - norm_cons);
    F2 = 0.3 * norm_ranks .* (1 + norm_cons);
    sigma = 0.2 * (1 + norm_cons);
    
    % Mutation components
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    perturbation = bsxfun(@times, sigma, randn(NP, D));
    
    % Combined mutation
    offspring = popdecs + bsxfun(@times, F1, elite_diff) + ...
                         bsxfun(@times, F2, rand_diff) + perturbation;
    
    % Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection with adaptive factor
    reflect_factor = 0.5 + 0.3 * norm_cons;
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
                (popdecs + reflect_factor .* (lb_rep - popdecs)) .* mask_low + ...
                (popdecs + reflect_factor .* (ub_rep - popdecs)) .* mask_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end