% MATLAB Code
function [offspring] = updateFunc569(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraint violations (0 to 1)
    abs_cons = abs(cons);
    norm_cons = abs_cons ./ (max(abs_cons) + eps);
    
    % Select elite individual
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
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
    
    % Generate random indices (different from current and each other)
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], r1(i))(randi(NP-2)), idx)';
    
    % Adaptive parameters
    F_base = 0.8 * (1 - norm_ranks);
    F_adapt = F_base .* (1 - norm_cons);
    sigma = 0.1 + 0.3 * norm_cons;
    
    % Mutation
    diff = popdecs(r1,:) - popdecs(r2,:);
    noise = sigma .* randn(NP, D);
    offspring = repmat(elite, NP, 1) + F_adapt .* diff + noise;
    
    % Boundary handling with random reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Lower bound violation
    mask_low = offspring < lb_rep;
    rand_factors = rand(NP, D);
    offspring = offspring .* ~mask_low + ...
        (popdecs + rand_factors .* (lb_rep - popdecs)) .* mask_low;
    
    % Upper bound violation
    mask_high = offspring > ub_rep;
    rand_factors = rand(NP, D);
    offspring = offspring .* ~mask_high + ...
        (popdecs + rand_factors .* (ub_rep - popdecs)) .* mask_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end