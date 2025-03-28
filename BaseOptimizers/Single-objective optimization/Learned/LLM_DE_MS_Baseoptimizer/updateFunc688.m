% MATLAB Code
function [offspring] = updateFunc688(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Feasibility-aware elite selection
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Compute adaptive scaling factors
    max_fit = max(popfits);
    min_fit = min(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    max_cons = max(abs(cons)) + eps;
    norm_cons = abs(cons) / max_cons;
    
    F = 0.5 * (1 + norm_cons) .* (1 - norm_fits);
    
    % 3. Compute direction vectors
    elite_dir = bsxfun(@minus, elite, popdecs);
    centroid = mean(popdecs, 1);
    centroid_dir = bsxfun(@minus, centroid, popdecs);
    
    % 4. Hybrid mutation with Gaussian perturbation
    rand_perturb = 0.1 * randn(NP, D) .* (ub - lb);
    mutant = popdecs + F .* (elite_dir + 0.7 * centroid_dir) + rand_perturb;
    
    % 5. Rank-based adaptive crossover
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
    
    % 6. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    offspring = min(max(offspring, lb_rep), ub_rep);
end