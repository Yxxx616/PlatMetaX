% MATLAB Code
function [offspring] = updateFunc697(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % 1. Adaptive scaling factors
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons));
    norm_f = (popfits - f_min) / (f_max - f_min + eps);
    cons_ratio = abs(cons) / (c_max + eps);
    
    F1 = 0.4 + 0.3 * norm_f + 0.3 * cons_ratio;
    F2 = 0.5 * (1 + cons_ratio);
    F3 = 0.8 - 0.4 * norm_f;
    
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
    elite_dir = bsxfun(@times, elite - popdecs, F1);
    
    cons_sign = sign(cons);
    cons_perturb = bsxfun(@times, popdecs(rand_idx1,:) - popdecs(rand_idx2,:), ...
                         cons_sign(:, ones(1,D))) .* F2(:, ones(1,D));
    
    % Fitness-weighted difference with rank-based weights
    [~, rank_order] = sort(penalty);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    w = 0.7 * (1 - ranks/NP);
    
    fit_diff = bsxfun(@times, best - popdecs, w(:, ones(1,D))) + ...
               bsxfun(@times, popdecs(rand_idx3,:) - popdecs(rand_idx4,:), ...
               1 - w(:, ones(1,D)));
    fit_diff = bsxfun(@times, fit_diff, F3(:, ones(1,D)));
    
    % 5. Combined mutation
    mutant = popdecs + elite_dir + cons_perturb + fit_diff;
    
    % 6. Adaptive crossover
    CR = 0.9 * (1 - ranks/NP).^0.5;
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