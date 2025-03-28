% MATLAB Code
function [offspring] = updateFunc844(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Elite selection
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Identify best and worst solutions
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    best = popdecs(best_idx, :);
    worst = popdecs(worst_idx, :);
    
    % Create direction vectors
    elite_rep = repmat(elite, NP, 1);
    v1 = elite_rep - popdecs;
    
    best_rep = repmat(best, NP, 1);
    worst_rep = repmat(worst, NP, 1);
    v2 = best_rep - worst_rep;
    
    % Random pairs for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    v3 = popdecs(r1,:) - popdecs(r2,:);
    
    % Adaptive weights
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    w_f = (popfits - f_min) / (f_max - f_min + eps);
    w_c = (cons - c_min) / (c_max - c_min + eps);
    
    % Adaptive parameters
    F = 0.5 + 0.3 * (1 - w_c) .* (1 + w_f);
    CR = 0.9 - 0.4 * w_f;
    
    % Mutation
    mutant = popdecs + F .* (v1 + 0.5 * v2 + 0.3 * v3);
    
    % Crossover
    mask = rand(NP, D) < CR(:, ones(1, D));
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
    offspring = max(min(offspring, ub_rep), lb_rep);
end