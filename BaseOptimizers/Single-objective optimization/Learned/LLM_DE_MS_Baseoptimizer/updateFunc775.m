% MATLAB Code
function [offspring] = updateFunc775(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify elite solution (best feasible or least infeasible)
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Identify extreme solutions
    [~, low_cons_idx] = min(cons);
    [~, high_cons_idx] = max(cons);
    [~, low_fit_idx] = min(popfits);
    [~, high_fit_idx] = max(popfits);
    
    x_low_cons = popdecs(low_cons_idx, :);
    x_high_cons = popdecs(high_cons_idx, :);
    x_low_fit = popdecs(low_fit_idx, :);
    x_high_fit = popdecs(high_fit_idx, :);
    
    % Normalize fitness and constraints to [0,1]
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    
    w_f = (popfits - f_min) / (f_max - f_min + eps);
    w_c = (cons - c_min) / (c_max - c_min + eps);
    
    % Generate random indices (vectorized)
    r1 = randi(NP-1, NP, 1);
    r1 = r1 + (r1 >= (1:NP)');
    r2 = randi(NP-2, NP, 1);
    r2 = r2 + (r2 >= min(r1, (1:NP)');
    r2 = r2 + (r2 >= max(r1, (1:NP)');
    
    % Compute direction vectors
    elite_rep = repmat(elite, NP, 1);
    low_cons_rep = repmat(x_low_cons, NP, 1);
    high_cons_rep = repmat(x_high_cons, NP, 1);
    low_fit_rep = repmat(x_low_fit, NP, 1);
    high_fit_rep = repmat(x_high_fit, NP, 1);
    
    v_elite = elite_rep - popdecs;
    v_cons = low_cons_rep - high_cons_rep;
    v_fit = low_fit_rep - high_fit_rep;
    v_rand = popdecs(r1,:) - popdecs(r2,:);
    
    % Adaptive weights
    alpha = 0.5 * (1 - w_f) + 0.2 * rand(NP, 1);
    beta = 0.3 * w_c + 0.2 * rand(NP, 1);
    gamma = 0.3 * (1 - w_c) + 0.2 * rand(NP, 1);
    delta = 0.3 * w_f + 0.2 * rand(NP, 1);
    
    % Combine components
    v = alpha .* v_elite + beta .* v_cons + gamma .* v_fit + delta .* v_rand;
    
    % Adaptive scaling factor
    F = 0.6 + 0.3 ./ (1 + exp(-5 * w_c));
    
    % Mutation
    mutant = popdecs + F .* v;
    
    % Adaptive crossover
    CR = 0.85 - 0.35 * w_c;
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