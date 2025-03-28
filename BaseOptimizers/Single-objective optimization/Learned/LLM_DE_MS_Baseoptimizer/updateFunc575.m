% MATLAB Code
function [offspring] = updateFunc575(popdecs, popfits, cons)
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
    
    % Fitness-based ranking with constraint consideration
    combined = popfits + norm_cons;
    [~, rank_order] = sort(combined);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    norm_ranks = ranks / NP;
    
    % Generate unique random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], r1(i))(randi(NP-2)), idx)';
    
    % Adaptive parameters
    F1 = 0.8 * (1 - norm_ranks) .* (1 - norm_cons);
    F2 = 0.6 * norm_ranks .* (1 + norm_cons);
    sigma = 0.2 * (1 + norm_cons);
    CR = 0.5 + 0.4 * (1 - norm_ranks) .* (1 - norm_cons);
    
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
    
    % Boundary handling - random reinitialization
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    out_of_bounds = offspring < lb_rep | offspring > ub_rep;
    rand_vals = rand(NP, D) .* (ub_rep - lb_rep) + lb_rep;
    offspring(out_of_bounds) = rand_vals(out_of_bounds);
end