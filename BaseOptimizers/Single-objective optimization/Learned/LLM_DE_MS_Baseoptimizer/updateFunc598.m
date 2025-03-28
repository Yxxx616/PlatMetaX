% MATLAB Code
function [offspring] = updateFunc598(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    cons_pos = max(0, cons);
    max_cons = max(cons_pos) + eps;
    cons_norm = cons_pos ./ max_cons;
    
    f_range = max(popfits) - min(popfits) + eps;
    f_norm = (popfits - min(popfits)) / f_range;
    
    % Feasibility handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons_pos);
        elite = popdecs(elite_idx, :);
    end
    
    % Get best and worst solutions
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    % Generate random indices (vectorized)
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    r3 = randi(NP, NP, 1);
    r4 = randi(NP, NP, 1);
    
    % Ensure distinct indices
    for i = 1:NP
        while r1(i) == i, r1(i) = randi(NP); end
        while r2(i) == i || r2(i) == r1(i), r2(i) = randi(NP); end
        while r3(i) == i || r3(i) == r1(i) || r3(i) == r2(i), r3(i) = randi(NP); end
        while r4(i) == i || r4(i) == r1(i) || r4(i) == r2(i) || r4(i) == r3(i), r4(i) = randi(NP); end
    end
    
    % Adaptive parameters with non-linear mapping
    F = 0.3 + 0.5 * exp(-2*cons_norm) + 0.2 * exp(-2*f_norm);
    CR = 0.1 + 0.6 * (1 - tanh(cons_norm)) + 0.3 * (1 - tanh(f_norm));
    
    % Vectorized mutation with enhanced components
    elite_rep = repmat(elite, NP, 1);
    best_rep = repmat(x_best, NP, 1);
    worst_rep = repmat(x_worst, NP, 1);
    
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    diff3 = best_rep - worst_rep;
    
    % Enhanced mutation with non-linear combination
    mutant = elite_rep + F .* diff1 + ...
             0.5 * (1 - cons_norm.^2) .* diff2 + ...
             0.5 * (f_norm.^2) .* diff3;
    
    % Crossover with jitter and adaptive CR
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | repmat((1:D) == j_rand, NP, 1);
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    % Adaptive reflection based on fitness
    reflect_factor = 1 + 0.5 * f_norm;
    offspring(below_lb) = (1 + reflect_factor(below_lb(:,1))) .* lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = (1 + reflect_factor(above_ub(:,1))) .* ub_rep(above_ub) - offspring(above_ub);
    
    % Final bounds enforcement with small perturbation
    offspring = min(max(offspring, lb_rep), ub_rep);
    perturb = 0.01 * (ub - lb) .* randn(NP, D);
    offspring = offspring + perturb;
    offspring = min(max(offspring, lb_rep), ub_rep);
end