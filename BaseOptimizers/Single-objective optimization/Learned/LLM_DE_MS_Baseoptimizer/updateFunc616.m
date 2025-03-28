% MATLAB Code
function [offspring] = updateFunc616(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Identify elite solution (best feasible or least infeasible)
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Identify best solution by fitness
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Normalize fitness and constraints
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    
    if f_max > f_min
        f_norm = (popfits - f_min) / (f_max - f_min);
    else
        f_norm = zeros(NP, 1);
    end
    
    if c_max > c_min
        c_norm = (cons - c_min) / (c_max - c_min);
    else
        c_norm = zeros(NP, 1);
    end
    
    % Adaptive parameters
    F = 0.5 + 0.3 * (1 - f_norm) .* (1 - c_norm);
    CR = 0.2 + 0.6 * (1 - sqrt(f_norm .* c_norm));
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randsample(setdiff(idx, i), 1), idx');
    r2 = arrayfun(@(i) randsample(setdiff(idx, [i, r1(i)]), 1), idx');
    
    % Mutation
    elite_rep = repmat(elite, NP, 1);
    best_rep = repmat(x_best, NP, 1);
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = best_rep - popdecs;
    mutant = elite_rep + F .* diff1 + (1 - F) .* diff2;
    
    % Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | repmat((1:D) == j_rand, NP, 1);
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with bounce-back
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = lb_rep(below_lb) + rand(sum(below_lb(:)),1) .* ...
        (popdecs(below_lb) - lb_rep(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - rand(sum(above_ub(:)),1) .* ...
        (ub_rep(above_ub) - popdecs(above_ub));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end