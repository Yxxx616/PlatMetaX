% MATLAB Code
function [offspring] = updateFunc605(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    cons_pos = max(0, cons);
    max_cons = max(cons_pos) + eps;
    cons_norm = cons_pos ./ max_cons;
    
    % Normalize fitness (inverted for minimization)
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    f_norm = (popfits - f_min) ./ f_range;
    
    % Identify elite solution (best feasible or least infeasible)
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons_pos);
        elite = popdecs(elite_idx, :);
    end
    
    % Get best solution based on fitness
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
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
    
    % Adaptive parameters
    F = 0.5 + 0.3 * rand(NP, 1);
    w_f = 0.5 * (1 + f_norm);
    w_c = 0.5 * (1 + cons_norm);
    
    % Vectorized mutation components
    elite_rep = repmat(elite, NP, 1);
    best_rep = repmat(x_best, NP, 1);
    
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    diff3 = best_rep - popdecs;
    
    % Core mutation equation
    mutant = elite_rep + F .* diff1 + ...
             w_f .* diff3 + ...
             w_c .* diff2;
    
    % Enhanced perturbation for highly constrained solutions
    high_constraint = w_c > 0.8;
    if any(high_constraint)
        idx = find(high_constraint);
        sigma = 0.2 * w_c(idx);
        mutant(idx,:) = elite_rep(idx,:) + sigma .* randn(length(idx), D) .* (ub - lb);
    end
    
    % Adaptive crossover with fitness and constraint awareness
    CR = 0.1 + 0.7 * (1 - cons_norm) .* (1 - f_norm);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | repmat((1:D) == j_rand, NP, 1);
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with reflection and clamping
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end