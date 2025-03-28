% MATLAB Code
function [offspring] = updateFunc693(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % 1. Adaptive scaling factor with fitness and constraint balance
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons));
    norm_f = (popfits - f_min) / (f_max - f_min + eps);
    cons_ratio = abs(cons) / (c_max + eps);
    F = 0.5 + 0.5 * norm_f .* (1 - cons_ratio);
    
    % 2. Select elite based on constrained fitness
    penalty = popfits + 1e6 * max(0, cons);
    [~, elite_idx] = min(penalty);
    elite = popdecs(elite_idx, :);
    
    % 3. Compute hybrid direction vectors
    % Elite direction
    elite_dir = bsxfun(@minus, elite, popdecs);
    
    % Constraint-aware random direction
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    rand_dir = popdecs(rand_idx1, :) - popdecs(rand_idx2, :);
    cons_sign = sign(cons);
    cons_dir = bsxfun(@times, rand_dir, cons_sign);
    
    % Weighted combination
    w = cons_ratio;
    direction = bsxfun(@times, elite_dir, w) + bsxfun(@times, cons_dir, 1-w);
    
    % 4. Hybrid mutation
    mutant = popdecs + bsxfun(@times, direction, F);
    
    % 5. Rank-based adaptive crossover
    [~, rank_order] = sort(penalty);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    CR = 0.9 * (1 - ranks/NP).^0.5;
    
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 6. Improved boundary repair
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    r = rand(NP,D);
    offspring(below_lb) = lb_rep(below_lb) + 0.5*r(below_lb).*(elite(below_lb) - lb_rep(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - 0.5*r(above_ub).*(ub_rep(above_ub) - elite(above_ub));
    
    % 7. Final clipping to ensure feasibility
    offspring = max(min(offspring, ub_rep), lb_rep);
end