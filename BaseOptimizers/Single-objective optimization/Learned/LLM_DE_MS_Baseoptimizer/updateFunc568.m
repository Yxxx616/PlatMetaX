% MATLAB Code
function [offspring] = updateFunc568(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints (0 to 1)
    norm_cons = abs(cons) ./ (max(abs(cons)) + eps);
    
    % Select elite based on feasible solutions
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + norm_cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Rank population (0=best, NP-1=worst)
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = 0:NP-1;
    norm_ranks = ranks / (NP-1);
    
    % Generate random indices (different from current)
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    
    % Adaptive scaling factors
    F1 = 0.7 * (1 - norm_ranks) .* (1 - norm_cons);
    F2 = 0.5 * norm_ranks .* (1 + norm_cons);
    sigma = 0.2 * (1 + norm_cons);
    
    % Mutation components
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    perturbation = sigma .* randn(NP, D);
    
    % Combined mutation
    offspring = popdecs + F1 .* elite_diff + F2 .* rand_diff + perturbation;
    
    % Adaptive boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection factor based on constraint violation
    reflect_factor = 0.5 * norm_cons;
    
    % Handle lower bounds
    mask_low = offspring < lb_rep;
    offspring = offspring .* ~mask_low + ...
        (popdecs + reflect_factor .* (lb_rep - popdecs)) .* mask_low;
    
    % Handle upper bounds
    mask_high = offspring > ub_rep;
    offspring = offspring .* ~mask_high + ...
        (popdecs + reflect_factor .* (ub_rep - popdecs)) .* mask_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end