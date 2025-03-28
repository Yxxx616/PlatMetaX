% MATLAB Code
function [offspring] = updateFunc114(popdecs, popfits, cons)
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
    x_best = elites(1,:); % Best individual
    
    % Generate mutation vectors (vectorized)
    candidates = setdiff(1:NP, elite_idx');
    r1 = candidates(randi(length(candidates), NP, 1));
    r2 = candidates(randi(length(candidates), NP, 1));
    r3 = candidates(randi(length(candidates), NP, 1));
    r4 = candidates(randi(length(candidates), NP, 1));
    
    % Adaptive scaling factor
    F = 0.4 + 0.4 * (1 - norm_fits) + 0.2 * norm_cons;
    F = repmat(F, 1, D);
    
    % Differential mutation
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    v = repmat(x_best, NP, 1) + F.*diff1 + (1-F).*diff2;
    
    % Dynamic crossover rate
    CR = 0.9 - 0.5 * norm_fits + 0.1 * norm_cons;
    CR = repmat(CR, 1, D);
    
    % Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Generate offspring
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % Boundary control with random reinitialization
    out_of_bounds = offspring < lb | offspring > ub;
    rand_vals = repmat(lb, NP, 1) + rand(NP, D) .* repmat(ub-lb, NP, 1);
    offspring(out_of_bounds) = rand_vals(out_of_bounds);
end