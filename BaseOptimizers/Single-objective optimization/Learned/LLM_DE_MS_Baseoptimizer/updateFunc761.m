% MATLAB Code
function [offspring] = updateFunc761(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify elite and best feasible solutions
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
        best_feas = elite;
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
        best_feas = elite;
    end
    
    % Identify extreme constraint solutions
    [~, low_cons_idx] = min(cons);
    [~, high_cons_idx] = max(cons);
    x_low_cons = popdecs(low_cons_idx, :);
    x_high_cons = popdecs(high_cons_idx, :);
    
    % Normalize fitness and constraints
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    
    w_f = (popfits - f_min) / (f_max - f_min + eps);
    w_c = (cons - c_min) / (c_max - c_min + eps);
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = (r1 == (1:NP)');
    r1(mask) = mod(r1(mask) + randi(NP-1, sum(mask), 1), NP) + 1;
    mask = (r2 == (1:NP)');
    r2(mask) = mod(r2(mask) + randi(NP-1, sum(mask), 1), NP) + 1;
    
    % Compute direction vectors
    elite_rep = repmat(elite, NP, 1);
    best_feas_rep = repmat(best_feas, NP, 1);
    low_cons_rep = repmat(x_low_cons, NP, 1);
    high_cons_rep = repmat(x_high_cons, NP, 1);
    
    v_elite = elite_rep - popdecs;
    v_feas = best_feas_rep - popdecs;
    v_rand = popdecs(r1,:) - popdecs(r2,:);
    v_cons = low_cons_rep - high_cons_rep;
    
    % Adaptive weights
    alpha = 0.5 * (1 - repmat(w_f, 1, D)) + 0.1 * rand(NP, D);
    beta = 0.3 * (1 - repmat(w_c, 1, D)) + 0.1 * rand(NP, D);
    gamma = 0.2 * repmat(w_c, 1, D) + 0.1 * rand(NP, D);
    delta = 0.2 * rand(NP, D);
    
    % Constraint-adaptive scale factor
    F = 0.6 + 0.2 * (1 - tanh(3 * repmat(w_c, 1, D)));
    
    % Combined mutation
    mutant = popdecs + F .* (alpha .* v_elite + beta .* v_feas + gamma .* v_cons + delta .* v_rand);
    
    % Constraint-adaptive crossover
    CR = 0.9 - 0.5 * repmat(w_c, 1, D);
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
    
    % Final projection
    offspring = max(min(offspring, ub_rep), lb_rep);
end