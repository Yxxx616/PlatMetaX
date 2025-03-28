% MATLAB Code
function [offspring] = updateFunc703(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % 1. Compute adaptive scaling factors
    f_min = min(popfits);
    f_max = max(popfits);
    c_min = min(cons);
    c_max = max(cons);
    
    norm_f = (popfits - f_min) / (f_max - f_min + eps);
    norm_c = (cons - c_min) / (c_max - c_min + eps);
    
    F = 0.4 + 0.4 * norm_c;
    G = 0.2 + 0.6 * norm_f;
    
    % 2. Select elite (considering constraints) and best (fitness only)
    penalty = popfits + 1e6 * max(0, cons);
    [~, elite_idx] = min(penalty);
    elite = popdecs(elite_idx, :);
    
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    % 3. Generate base vectors
    use_best = rand(NP,1) < 0.7;
    base = repmat(elite, NP, 1);
    base(use_best,:) = repmat(best, sum(use_best), 1);
    
    % 4. Generate random indices for differential vectors
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    
    % 5. Compute direction vectors
    cons_diff = cons(rand_idx1) - cons(rand_idx2);
    cons_sign = sign(cons_diff);
    cons_dir = popdecs(rand_idx1,:) - popdecs(rand_idx2,:);
    cons_dir = cons_dir .* cons_sign(:, ones(1,D));
    
    fit_dir = best - popdecs;
    no_cons = (cons <= 0);
    dir = cons_dir + fit_dir .* (1 - no_cons(:, ones(1,D)));
    
    % 6. Combined mutation
    mutant = base + dir .* F(:, ones(1,D)) + ...
             (best - elite) .* G(:, ones(1,D));
    
    % 7. Rank-based adaptive crossover
    [~, rank_order] = sort(penalty);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    CR = 0.9 * (1 - ranks/NP).^0.5;
    
    mask = rand(NP,D) < CR(:, ones(1,D));
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
    
    % Final clipping to ensure feasibility
    offspring = max(min(offspring, ub_rep), lb_rep);
end