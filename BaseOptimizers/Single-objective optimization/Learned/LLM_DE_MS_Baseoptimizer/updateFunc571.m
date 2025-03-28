% MATLAB Code
function [offspring] = updateFunc571(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraint violations
    abs_cons = abs(cons);
    norm_cons = abs_cons ./ (max(abs_cons) + eps);
    
    % Elite selection (best feasible or least infeasible)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + norm_cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Fitness ranking (0=best, NP-1=worst)
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = 0:NP-1;
    norm_ranks = ranks / (NP-1);
    
    % Generate unique random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], r1(i))(randi(NP-2)), idx)';
    
    % Adaptive parameters
    F = 0.4 + 0.4 * (1 - norm_ranks) .* (1 - norm_cons);
    sigma = 0.3 * (1 + norm_cons);
    
    % Mutation with elite guidance
    diff = popdecs(r1,:) - popdecs(r2,:);
    noise = sigma .* randn(NP, D);
    offspring = repmat(elite, NP, 1) + F .* diff + noise;
    
    % Opposition-based boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Lower bound violation
    mask_low = offspring < lb_rep;
    offspring(mask_low) = lb_rep(mask_low) + (ub_rep(mask_low) - offspring(mask_low));
    
    % Upper bound violation
    mask_high = offspring > ub_rep;
    offspring(mask_high) = ub_rep(mask_high) - (offspring(mask_high) - lb_rep(mask_high));
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end