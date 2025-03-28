% MATLAB Code
function [offspring] = updateFunc826(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify elite solution
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Identify best/worst by fitness
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    % Identify most/least feasible
    [~, feas_idx] = min(cons);
    [~, infeas_idx] = max(cons);
    x_feas = popdecs(feas_idx, :);
    x_infeas = popdecs(infeas_idx, :);
    
    % Normalize fitness and constraints
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    w_f = (popfits - f_min) / (f_max - f_min + eps);
    w_c = (cons - c_min) / (c_max - c_min + eps);
    
    % Create direction vectors (vectorized)
    elite_rep = repmat(elite, NP, 1);
    best_rep = repmat(x_best, NP, 1);
    worst_rep = repmat(x_worst, NP, 1);
    feas_rep = repmat(x_feas, NP, 1);
    infeas_rep = repmat(x_infeas, NP, 1);
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Opposition-based direction
    v_opp = (lb_rep + ub_rep - popdecs) - popdecs;
    
    % Random pairs for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    rand_rep1 = popdecs(r1, :);
    rand_rep2 = popdecs(r2, :);
    
    % Compute direction vectors
    v_elite = elite_rep - popdecs;
    v_feas = feas_rep - infeas_rep;
    v_grad = (best_rep - worst_rep) .* (1 + w_f);
    v_rand = rand_rep1 - rand_rep2;
    
    % Adaptive weights
    alpha = 0.7 * (1 - tanh(w_c));
    beta = 0.2 * tanh(w_c);
    gamma = 0.1 * (1 + sin(pi * w_f));
    delta = 0.05 * (1 - w_c);
    eta = 0.05 * w_c;
    
    % Combined mutation vector with adaptive F
    F = 0.6 + 0.2 * (1 - w_c);
    v = alpha .* v_elite + beta .* v_feas + gamma .* v_grad + delta .* v_rand + eta .* v_opp;
    mutant = popdecs + F .* v;
    
    % Adaptive crossover
    CR = 0.95 - 0.45 * w_f;
    mask = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with reflection
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    offspring = max(min(offspring, ub_rep), lb_rep);
end