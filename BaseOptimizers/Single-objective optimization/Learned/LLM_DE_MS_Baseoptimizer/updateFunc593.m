% MATLAB Code
function [offspring] = updateFunc593(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    cons_pos = max(0, cons);
    max_cons = max(cons_pos) + eps;
    cons_norm = cons_pos ./ max_cons;
    
    % Feasibility-based elite selection
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons_pos);
        elite = popdecs(elite_idx, :);
    end
    
    % Best and worst based on fitness
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    % Generate random indices
    idx = 1:NP;
    r1 = zeros(NP, 1); r2 = zeros(NP, 1); r3 = zeros(NP, 1);
    for i = 1:NP
        available = setdiff(idx, i);
        r1(i) = available(randi(NP-1));
        remaining = setdiff(available, r1(i));
        r2(i) = remaining(randi(NP-2));
        remaining = setdiff(remaining, r2(i));
        r3(i) = remaining(randi(NP-3));
    end
    
    % Adaptive parameters
    f_norm = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    F = 0.2 + 0.6 * (1 - cons_norm) + 0.2 * f_norm;
    CR = 0.1 + 0.8 * (1 - cons_norm);
    
    % Hybrid mutation
    elite_rep = repmat(elite, NP, 1);
    best_rep = repmat(x_best, NP, 1);
    worst_rep = repmat(x_worst, NP, 1);
    diff1 = popdecs(r1,:) - popdecs;
    diff2 = popdecs(r2,:) - popdecs(r3,:);
    diff3 = best_rep - worst_rep;
    mutant = elite_rep + F .* diff1 + F .* diff2 + 0.5 * F .* diff3;
    
    % Crossover with jitter
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | repmat((1:D) == j_rand, NP, 1);
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling - bounce back
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = lb_rep(below_lb) + rand(NP,D) .* (popdecs(below_lb) - lb_rep(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - rand(NP,D) .* (ub_rep(above_ub) - popdecs(above_ub));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end