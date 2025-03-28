% MATLAB Code
function [offspring] = updateFunc115(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    c_min = min(cons);
    c_max = max(cons);
    norm_cons = (cons - c_min) / (c_max - c_min + eps);
    
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    % Combined score for selection (lower is better)
    w1 = 0.7; w2 = 0.3;
    scores = w1 * norm_fits + w2 * norm_cons;
    [~, sorted_idx] = sort(scores);
    elite_idx = sorted_idx(1:3);
    elites = popdecs(elite_idx, :);
    x_best = elites(1,:);
    
    % Generate mutation vectors (vectorized)
    candidates = setdiff(1:NP, elite_idx');
    r1 = candidates(randi(length(candidates), NP, 1));
    r2 = candidates(randi(length(candidates), NP, 1));
    r3 = candidates(randi(length(candidates), NP, 1));
    r4 = candidates(randi(length(candidates), NP, 1));
    
    % Adaptive scaling factor
    F = 0.5 + 0.3 * (1 - norm_fits) + 0.2 * norm_cons;
    F = repmat(F, 1, D);
    
    % Differential mutation with elite guidance
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    v = repmat(x_best, NP, 1) + F.*diff1 + (1-F).*diff2;
    
    % Dynamic crossover rate
    CR = 0.9 - 0.4 * norm_fits + 0.2 * norm_cons;
    CR = repmat(CR, 1, D);
    
    % Crossover with jitter
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Generate offspring
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % Boundary control with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final boundary check (fallback to random if still out of bounds)
    out_of_bounds = offspring < lb_rep | offspring > ub_rep;
    rand_vals = lb_rep + rand(NP, D) .* (ub_rep - lb_rep);
    offspring(out_of_bounds) = rand_vals(out_of_bounds);
end