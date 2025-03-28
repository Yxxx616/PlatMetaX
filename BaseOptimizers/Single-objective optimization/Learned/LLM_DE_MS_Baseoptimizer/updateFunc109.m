% MATLAB Code
function [offspring] = updateFunc109(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    c_abs = abs(cons);
    c_max = max(c_abs);
    norm_cons = c_abs / (c_max + eps);
    
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    % Select elite individuals (top 3)
    beta = 0.8;
    elite_scores = popfits + beta * c_abs;
    [~, elite_idx] = sort(elite_scores);
    elite_idx = elite_idx(1:3);
    elites = popdecs(elite_idx, :);
    elite_scores = elite_scores(elite_idx);
    
    % Weighted mean of elites
    weights = (1./elite_scores) / sum(1./elite_scores);
    x_base = weights' * elites;
    
    % Opposition point
    x_opp = lb + ub - x_base;
    
    % Generate mutation vectors (vectorized)
    candidates = setdiff(1:NP, elite_idx');
    r1 = candidates(randi(length(candidates), NP, 1));
    r2 = candidates(randi(length(candidates), NP, 1));
    r3 = candidates(randi(length(candidates), NP, 1));
    
    % Adaptive scaling factor
    F = 0.5 + 0.3 * (1 - norm_fits) + 0.2 * norm_cons;
    F = repmat(F, 1, D);
    
    % Mutation
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - repmat(x_opp, NP, 1);
    v = repmat(x_base, NP, 1) + F.*diff1 + (1-F).*diff2;
    
    % Adaptive crossover rate
    CR = 0.6 + 0.3 * (1 - norm_fits);
    CR = repmat(CR, 1, D);
    
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Generate offspring
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % Boundary control with reflection
    below_lb = offspring < lb;
    above_ub = offspring > ub;
    offspring(below_lb) = 2*lb(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub(above_ub) - offspring(above_ub);
    
    % Final clamping
    offspring = min(max(offspring, lb), ub);
end