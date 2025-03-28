% MATLAB Code
function [offspring] = updateFunc686(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection based on feasibility
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Compute adaptive weights
    max_fit = max(popfits);
    min_fit = min(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    max_cons = max(abs(cons)) + eps;
    norm_cons = abs(cons) / max_cons;
    
    weights = (1 - norm_fits) .* (1 - norm_cons);
    weights = weights / sum(weights);
    
    % 3. Generate direction vectors
    elite_dir = repmat(elite, NP, 1) - popdecs;
    
    % Weighted centroid direction (vectorized)
    weighted_diff = zeros(NP, D);
    for i = 1:NP
        weighted_diff(i,:) = sum(bsxfun(@times, popdecs - popdecs(i,:), weights'), 1);
    end
    
    % 4. Adaptive scaling factors
    F_base = 0.5;
    F_var = 0.3;
    F = F_base + F_var * randn(NP, 1) .* (1 - norm_fits .* norm_cons);
    
    % 5. Random perturbation
    alpha = 0.1;
    rand_perturb = alpha * (rand(NP, D) - 0.5) .* (ub - lb);
    
    % 6. Enhanced mutation
    mutant = popdecs + F .* (elite_dir + weighted_diff) + rand_perturb;
    
    % 7. Rank-based adaptive crossover
    penalty = popfits + 1e6 * max(0, cons);
    [~, rank_order] = sort(penalty);
    ranks = zeros(NP, 1);
    ranks(rank_order) = 1:NP;
    CR = 0.9 * (1 - ranks/NP);
    
    mask = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 8. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    offspring = min(max(offspring, lb_rep), ub_rep);
end