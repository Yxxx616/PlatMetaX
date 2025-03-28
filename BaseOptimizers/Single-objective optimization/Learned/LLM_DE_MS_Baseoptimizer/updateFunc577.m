% MATLAB Code
function [offspring] = updateFunc577(popdecs, popfits, cons)
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
    
    % Rank population based on combined fitness and constraints
    combined = popfits + norm_cons;
    [~, rank_order] = sort(combined);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    norm_ranks = ranks / NP;  % Normalized to [0,1]
    
    % Generate unique random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff([idx(1:i-1) idx(i+1:end)], r1(i))(randi(NP-2)), idx)';
    
    % Adaptive parameters
    sqrt_ranks = sqrt(norm_ranks);
    F_base = 0.6 * (1 - norm_cons);
    w_rank = 0.5 * (1 - sqrt_ranks);
    sigma = 0.2 * (1 + norm_cons);
    
    % Mutation
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    noise = sigma .* randn(NP, D);
    mutant = popdecs + F_base .* elite_diff + w_rank .* rand_diff + noise;
    
    % Crossover with adaptive CR
    CR = 0.3 + 0.5 * (1 - norm_ranks) .* (1 - norm_cons);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) <= CR(:, ones(1,D)) | (1:D) == j_rand;
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling - bounce back
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    lower_violation = offspring < lb_rep;
    upper_violation = offspring > ub_rep;
    offspring(lower_violation) = lb_rep(lower_violation) + abs(offspring(lower_violation) - lb_rep(lower_violation));
    offspring(upper_violation) = ub_rep(upper_violation) - abs(offspring(upper_violation) - ub_rep(upper_violation));
    
    % Final clipping to ensure bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end