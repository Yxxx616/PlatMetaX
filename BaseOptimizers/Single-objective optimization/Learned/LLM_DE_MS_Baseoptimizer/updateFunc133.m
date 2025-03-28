% MATLAB Code
function [offspring] = updateFunc133(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_min = min(cons);
    c_max = max(cons);
    norm_cons = (cons - c_min) / (c_max - c_min + eps);
    
    % Select elite pool (top 20%)
    [~, sorted_idx] = sort(popfits);
    elite_size = max(2, floor(0.2*NP));
    elite_idx = sorted_idx(1:elite_size);
    best_idx = sorted_idx(1);
    
    % Create weighted elite center
    weights = 1./(1 + abs(popfits(elite_idx)) + abs(cons(elite_idx)));
    weights = weights / sum(weights);
    x_elite = weights' * popdecs(elite_idx,:);
    
    % Generate direction vectors
    non_elite = setdiff(1:NP, elite_idx);
    r1 = non_elite(randi(length(non_elite), NP, 1));
    r2 = non_elite(randi(length(non_elite), NP, 1));
    
    d_imp = repmat(popdecs(best_idx,:) - x_elite, NP, 1);
    d_div = popdecs(r1,:) - popdecs(r2,:);
    
    % Calculate adaptive coefficients
    F1 = 0.5 * (1 - norm_fits);
    F2 = 0.3 * (1 - norm_cons);
    sigma = 0.1 * (1 + norm_cons);
    
    % Create mutation vectors
    v = repmat(x_elite, NP, 1) + ...
        d_imp .* repmat(F1, 1, D) + ...
        d_div .* repmat(F2, 1, D) + ...
        sigma .* randn(NP, D);
    
    % Adaptive crossover rates
    CR = 0.7 + 0.2 * (1 - norm_fits) .* (1 - norm_cons);
    CR = repmat(CR, 1, D);
    
    % Crossover with guaranteed jitter
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Generate offspring
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % Boundary control with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    reflect_factor = 0.3 + 0.5 * repmat(norm_fits, 1, D) .* rand(NP, D);
    
    offspring(below_lb) = lb_rep(below_lb) + reflect_factor(below_lb) .* ...
                        (ub_rep(below_lb) - lb_rep(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - reflect_factor(above_ub) .* ...
                        (ub_rep(above_ub) - lb_rep(above_ub));
    
    % Final boundary check
    out_of_bounds = offspring < lb_rep | offspring > ub_rep;
    rand_vals = lb_rep + rand(NP, D) .* (ub_rep - lb_rep);
    offspring(out_of_bounds) = rand_vals(out_of_bounds);
end