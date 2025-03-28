% MATLAB Code
function [offspring] = updateFunc690(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute adaptive scaling factors
    sigma_f = std(popfits);
    mean_f = mean(popfits);
    c_max = max(abs(cons));
    F_base = 0.5 + 0.5 * tanh((popfits - mean_f) / sigma_f) .* (1 - abs(cons) / max(1e-6, c_max));
    
    % 2. Select elite based on constrained fitness
    penalty = popfits + 1e6 * max(0, cons);
    [~, elite_idx] = min(penalty);
    elite = popdecs(elite_idx, :);
    
    % 3. Compute direction vectors
    centroid = mean(popdecs, 1);
    sigma_c = std(cons);
    alpha = exp(-abs(cons) / max(1e-6, sigma_c));
    beta = 1 - alpha;
    gamma = abs(popfits) / max(abs(popfits));
    
    elite_dir = bsxfun(@minus, elite, popdecs);
    centroid_dir = bsxfun(@minus, centroid, popdecs);
    rand_dir = randn(NP, D);
    
    direction = alpha.*elite_dir + beta.*centroid_dir + gamma.*rand_dir;
    
    % 4. Constraint-aware mutation
    lambda = 0.2;
    cons_factor = 1 + lambda * abs(cons);
    mutant = popdecs + F_base(:, ones(1,D)) .* direction .* cons_factor(:, ones(1,D));
    
    % 5. Rank-based exponential crossover
    [~, rank_order] = sort(penalty);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    CR = 0.9 * (1 - ranks/NP).^1.5;
    
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 6. Smart boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    r = rand(NP,D);
    offspring(below_lb) = lb_rep(below_lb) + r(below_lb).*(popdecs(below_lb) - lb_rep(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - r(above_ub).*(ub_rep(above_ub) - popdecs(above_ub));
end