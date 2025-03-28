% MATLAB Code
function [offspring] = updateFunc699(popdecs, popfits, cons)
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
    
    F1 = 0.4 + 0.3 * norm_f;
    F2 = 0.2 + 0.5 * cons_ratio;
    F3 = 1 - F1;
    F4 = F2;
    
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
    cons_dir = bsxfun(@times, popdecs(rand_idx1,:) - popdecs(rand_idx2,:), ...
               cons_sign(:, ones(1,D))) .* F2(:, ones(1,D));
    
    fit_dir = bsxfun(@times, best - popdecs, F3) + ...
              bsxfun(@times, popdecs(rand_idx3,:) - popdecs(rand_idx4,:), ...
              F4(:, ones(1,D)));
    
    % 5. Combined mutation
    mutant = popdecs + elite_dir + cons_dir + fit_dir;
    
    % 6. Rank-based adaptive crossover
    [~, rank_order] = sort(penalty);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    CR = 0.85 * (1 - ranks/NP).^0.7;
    
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