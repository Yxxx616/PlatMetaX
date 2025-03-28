% MATLAB Code
function [offspring] = updateFunc573(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraint violations
    abs_cons = abs(cons);
    norm_cons = abs_cons ./ (max(abs_cons) + eps);
    
    % Elite selection
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + norm_cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Fitness ranking
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = 0:NP-1;
    norm_ranks = ranks / (NP-1);
    
    % Generate unique random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], r1(i))(randi(NP-2)), idx)';
    
    % Adaptive parameters
    F = 0.5 * (1 + norm_ranks) .* (1 - norm_cons);
    CR = 0.9 * (1 - norm_ranks) .* (1 - norm_cons);
    sigma = 0.2 * (1 + norm_cons);
    
    % Mutation
    diff = popdecs(r1,:) - popdecs(r2,:);
    elite_diff = repmat(elite, NP, 1) - popdecs;
    noise = sigma .* randn(NP, D);
    mutant = repmat(elite, NP, 1) + F .* diff + sigma .* elite_diff + noise;
    
    % Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) <= CR | (1:D) == j_rand;
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Lower bound violation
    mask_low = offspring < lb_rep;
    offspring(mask_low) = (lb_rep(mask_low) + popdecs(mask_low)) / 2;
    
    % Upper bound violation
    mask_high = offspring > ub_rep;
    offspring(mask_high) = (ub_rep(mask_high) + popdecs(mask_high)) / 2;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end