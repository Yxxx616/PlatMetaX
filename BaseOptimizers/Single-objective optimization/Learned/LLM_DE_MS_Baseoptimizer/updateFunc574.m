% MATLAB Code
function [offspring] = updateFunc574(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraint violations
    abs_cons = abs(cons);
    norm_cons = abs_cons ./ (max(abs_cons) + eps);
    
    % Elite selection - best feasible or least infeasible
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + norm_cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Fitness-based ranking
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;  % Higher rank = better
    norm_ranks = ranks / NP;
    
    % Generate unique random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], r1(i))(randi(NP-2)), idx)';
    
    % Adaptive parameters
    F1 = 0.7 * (1 - norm_ranks) .* (1 - norm_cons);
    F2 = 0.5 * norm_ranks .* (1 + norm_cons);
    sigma = 0.3 * (1 + norm_cons);
    CR = 0.9 * (1 - norm_ranks) .* (1 - norm_cons);
    
    % Mutation
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    noise = sigma .* randn(NP, D);
    mutant = popdecs + F1 .* elite_diff + F2 .* rand_diff + noise;
    
    % Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) <= CR | (1:D) == j_rand;
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling - reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Lower bound violation
    mask_low = offspring < lb_rep;
    offspring(mask_low) = 2*lb_rep(mask_low) - offspring(mask_low);
    
    % Upper bound violation
    mask_high = offspring > ub_rep;
    offspring(mask_high) = 2*ub_rep(mask_high) - offspring(mask_high);
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Additional check for extreme cases
    recheck = offspring < lb_rep | offspring > ub_rep;
    offspring(recheck) = lb_rep(recheck) + rand(sum(recheck(:)),1) .* ...
                        (ub_rep(recheck) - lb_rep(recheck));
end