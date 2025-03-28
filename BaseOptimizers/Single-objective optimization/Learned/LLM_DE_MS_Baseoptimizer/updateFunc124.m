% MATLAB Code
function [offspring] = updateFunc124(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Enhanced normalization with protection
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_min = min(cons);
    c_max = max(cons);
    norm_cons = (cons - c_min) / (c_max - c_min + eps);
    
    % Combined weights with exponential scaling
    weights = exp(-(norm_fits + norm_cons));
    
    % Select top 30% solutions for elite pool
    [~, sorted_idx] = sort(weights, 'descend');
    elite_size = max(3, floor(0.3*NP));
    elite_idx = sorted_idx(1:elite_size);
    
    % Weighted elite center
    elite_weights = weights(elite_idx) / sum(weights(elite_idx));
    x_best = elite_weights' * popdecs(elite_idx,:);
    
    % Adaptive scaling factors with non-linear combination
    F = 0.4 + 0.4 * (1 - norm_fits).^2 .* (1 - norm_cons).^2;
    F = repmat(F, 1, D);
    
    % Generate mutation vectors (vectorized)
    candidates = setdiff(1:NP, elite_idx');
    r1 = candidates(randi(length(candidates), NP, 1));
    r2 = candidates(randi(length(candidates), NP, 1));
    r3 = candidates(randi(length(candidates), NP, 1));
    r4 = candidates(randi(length(candidates), NP, 1));
    r5 = candidates(randi(length(candidates), NP, 1));
    r6 = candidates(randi(length(candidates), NP, 1));
    
    % Multi-directional differential mutation
    diff1 = popdecs(r1,:) - popdecs(r2,:) + popdecs(r3,:) - popdecs(r4,:);
    diff2 = popdecs(r5,:) - popdecs(r6,:);
    v = repmat(x_best, NP, 1) + F .* diff1 + 0.5*F .* diff2;
    
    % Dynamic crossover rates
    CR = 0.85 - 0.5 * norm_cons;
    CR = repmat(CR, 1, D);
    
    % Crossover with jitter
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Generate offspring
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % Enhanced boundary control
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    % Quality-based reflection with non-linear mapping
    reflect_factor = 0.2 + 0.6 * repmat(weights, 1, D).^2;
    offspring(below_lb) = lb_rep(below_lb) + reflect_factor(below_lb) .* ...
                         (lb_rep(below_lb) - offspring(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - reflect_factor(above_ub) .* ...
                         (offspring(above_ub) - ub_rep(above_ub));
    
    % Final boundary check with random reset
    out_of_bounds = offspring < lb_rep | offspring > ub_rep;
    rand_vals = lb_rep + rand(NP, D) .* (ub_rep - lb_rep);
    offspring(out_of_bounds) = rand_vals(out_of_bounds);
end