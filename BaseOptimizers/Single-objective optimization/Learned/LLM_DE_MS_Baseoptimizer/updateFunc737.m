% MATLAB Code
function [offspring] = updateFunc737(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify elite and best feasible solutions
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
        [~, best_feas_idx] = min(popfits(feasible));
        best_feas = popdecs(feasible(best_feas_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
        best_feas = elite;
    end
    
    % Normalize fitness and constraints
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    
    w_f = (popfits - f_min) / (f_max - f_min + eps);
    w_c = (cons - c_min) / (c_max - c_min + eps);
    
    % Generate random indices for diversity component
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    
    % Compute direction vectors
    elite_rep = repmat(elite, NP, 1);
    best_feas_rep = repmat(best_feas, NP, 1);
    
    v_elite = elite_rep - popdecs;
    v_feas = best_feas_rep - popdecs;
    v_div = popdecs(r1,:) - popdecs(r2,:);
    
    % Adaptive weights
    F = 0.6;
    alpha = 0.7 * (1 - repmat(w_f, 1, D));
    beta = 0.5 * repmat(w_c, 1, D);
    gamma = 0.3 * (1 - repmat(w_c, 1, D));
    
    % Combined mutation
    mutant = popdecs + F .* (alpha .* v_elite + beta .* v_feas + gamma .* v_div);
    
    % Adaptive crossover
    CR = 0.8 + 0.1 * repmat(w_c, 1, D);
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final projection to bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end