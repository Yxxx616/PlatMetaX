% MATLAB Code
function [offspring] = updateFunc747(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify elite and best feasible solutions
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
        best_feas = elite; % Same as elite in this case
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
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = (r1 == r2);
    while any(mask)
        r2(mask) = randi(NP, sum(mask), 1);
        mask = (r1 == r2);
    end
    
    % Compute direction vectors
    elite_rep = repmat(elite, NP, 1);
    best_feas_rep = repmat(best_feas, NP, 1);
    
    v_elite = elite_rep - popdecs;
    v_feas = best_feas_rep - popdecs;
    v_rand = popdecs(r1,:) - popdecs(r2,:);
    
    % Adaptive weights and scale factor
    alpha = 0.6 * (1 - repmat(w_f, 1, D)) + 0.2 * rand(NP, D);
    beta = 0.4 * (1 - repmat(w_c, 1, D)) + 0.1 * rand(NP, D);
    gamma = 0.2 * rand(NP, D);
    F = min(max(0.5 + 0.15 * randn(NP, 1), 0.8), 0.3);
    
    % Combined mutation
    mutant = popdecs + F .* (alpha .* v_elite + beta .* v_feas + gamma .* v_rand);
    
    % Adaptive crossover
    CR = 0.85 - 0.35 * repmat(w_c, 1, D);
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with reflection/projection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    reflect_mask = rand(NP, D) < 0.5;
    reflect_below = below_lb & reflect_mask;
    reflect_above = above_ub & reflect_mask;
    
    offspring(reflect_below) = 2*lb_rep(reflect_below) - offspring(reflect_below);
    offspring(reflect_above) = 2*ub_rep(reflect_above) - offspring(reflect_above);
    
    % Final projection for non-reflected solutions
    offspring = max(min(offspring, ub_rep), lb_rep);
end