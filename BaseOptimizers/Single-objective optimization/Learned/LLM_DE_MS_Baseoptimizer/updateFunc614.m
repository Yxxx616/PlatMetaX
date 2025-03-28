% MATLAB Code
function [offspring] = updateFunc614(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints
    cons_pos = max(0, cons);
    c_min = min(cons_pos);
    c_max = max(cons_pos);
    if c_max > c_min
        c_norm = (cons_pos - c_min) / (c_max - c_min);
    else
        c_norm = zeros(NP, 1);
    end
    
    % Normalize fitness
    f_min = min(popfits);
    f_max = max(popfits);
    if f_max > f_min
        f_norm = (popfits - f_min) / (f_max - f_min);
    else
        f_norm = zeros(NP, 1);
    end
    
    % Identify elite and best solutions
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons_pos);
        elite = popdecs(elite_idx, :);
    end
    
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randsample(setdiff(idx, i), 1), idx');
    r2 = arrayfun(@(i) randsample(setdiff(idx, [i, r1(i)]), 1), idx');
    r3 = arrayfun(@(i) randsample(setdiff(idx, [i, r1(i), r2(i)]), 1), idx');
    r4 = arrayfun(@(i) randsample(setdiff(idx, [i, r1(i), r2(i), r3(i)]), 1), idx');
    r5 = arrayfun(@(i) randsample(setdiff(idx, [i, r1(i), r2(i), r3(i), r4(i)]), 1), idx');
    
    % Adaptive parameters
    F = 0.5 * (1 + rand(NP, 1)) .* (1 - f_norm);
    alpha = 0.5 * (1 + c_norm);
    
    % Vectorized mutation
    elite_rep = repmat(elite, NP, 1);
    best_rep = repmat(x_best, NP, 1);
    
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = best_rep - popdecs(r3,:);
    diff3 = popdecs(r4,:) - popdecs(r5,:);
    
    mutant = elite_rep + F .* diff1 + ...
             alpha .* diff2 + ...
             (1-alpha) .* diff3;
    
    % Adaptive crossover
    CR = 0.9 - 0.5 * (f_norm + c_norm);
    CR = min(max(CR, 0.1), 0.9);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | repmat((1:D) == j_rand, NP, 1);
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end