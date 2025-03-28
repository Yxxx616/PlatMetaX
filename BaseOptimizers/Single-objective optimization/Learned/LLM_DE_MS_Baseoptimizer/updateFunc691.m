% MATLAB Code
function [offspring] = updateFunc691(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % 1. Compute adaptive scaling factors
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons));
    norm_f = (popfits - f_min) / (f_max - f_min + eps);
    cons_ratio = abs(cons) / (c_max + eps);
    F = 0.4 + 0.6 * norm_f .* (1 - cons_ratio);
    
    % 2. Select elite based on constrained fitness
    penalty = popfits + 1e6 * max(0, cons);
    [~, elite_idx] = min(penalty);
    elite = popdecs(elite_idx, :);
    
    % 3. Compute direction vectors
    % Elite direction
    elite_dir = bsxfun(@minus, elite, popdecs);
    
    % Constraint-violation direction
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    rand_dir = popdecs(rand_idx1, :) - popdecs(rand_idx2, :);
    cons_sign = sign(cons);
    cons_dir = bsxfun(@times, rand_dir, cons_sign);
    
    % Weight for direction combination
    w = cons_ratio;
    direction = bsxfun(@times, elite_dir, 1-w) + bsxfun(@times, cons_dir, w);
    
    % 4. Constraint-aware mutation
    mutant = popdecs + bsxfun(@times, direction, F);
    
    % 5. Rank-based exponential crossover
    [~, rank_order] = sort(penalty);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    CR = 0.85 * (1 - ranks/NP).^1.2;
    
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