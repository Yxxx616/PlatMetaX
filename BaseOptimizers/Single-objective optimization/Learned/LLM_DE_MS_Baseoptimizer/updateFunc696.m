% MATLAB Code
function [offspring] = updateFunc696(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % 1. Adaptive scaling factor combining fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons));
    norm_f = (popfits - f_min) / (f_max - f_min + eps);
    cons_ratio = abs(cons) / (c_max + eps);
    F = 0.5 + 0.3 * norm_f + 0.2 * cons_ratio;
    
    % 2. Select elite (considering constraints) and best (fitness only)
    penalty = popfits + 1e6 * max(0, cons);
    [~, elite_idx] = min(penalty);
    elite = popdecs(elite_idx, :);
    
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    % 3. Generate random indices for differential vectors
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    rand_idx3 = randi(NP, NP, 1);
    rand_idx4 = randi(NP, NP, 1);
    
    % 4. Compute direction vectors
    elite_dir = elite - popdecs;
    cons_perturb = (popdecs(rand_idx1,:) - popdecs(rand_idx2,:)) .* sign(cons);
    
    % Fitness-weighted difference
    w_f = 0.7 * norm_f;
    fit_diff = bsxfun(@times, best - popdecs, w_f) + ...
               bsxfun(@times, popdecs(rand_idx3,:) - popdecs(rand_idx4,:), 1-w_f);
    
    % 5. Hybrid mutation with adaptive scaling
    mutant = popdecs + bsxfun(@times, elite_dir + 0.6*cons_perturb + 0.4*fit_diff, F);
    
    % 6. Rank-based adaptive crossover
    [~, rank_order] = sort(penalty);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    CR = 0.85 * (1 - ranks/NP).^0.4;
    
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 7. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final clipping to ensure feasibility
    offspring = max(min(offspring, ub_rep), lb_rep);
end