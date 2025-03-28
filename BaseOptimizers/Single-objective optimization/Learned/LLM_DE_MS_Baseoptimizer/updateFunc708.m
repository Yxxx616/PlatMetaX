% MATLAB Code
function [offspring] = updateFunc708(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % 1. Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    c_min = min(cons);
    c_max = max(cons);
    
    norm_f = (popfits - f_min) / (f_max - f_min + eps);
    norm_c = (cons - c_min) / (c_max - c_min + eps);
    
    % 2. Compute adaptive scaling factors
    F = 0.3 + 0.4 * norm_c;
    G = 0.5 * (1 - (1:NP)'/NP).^0.3;
    H = 0.2 * norm_f;
    
    % 3. Select elite (best feasible) and best (overall)
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + 1e6 * max(0, cons));
        elite = popdecs(elite_idx, :);
    end
    
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    % 4. Generate random indices for differential vectors
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    
    % 5. Compute constraint-aware direction
    cons_diff = cons(rand_idx1) - cons(rand_idx2);
    cons_sign = sign(cons_diff);
    cons_dir = popdecs(rand_idx1,:) - popdecs(rand_idx2,:);
    cons_dir = cons_dir .* cons_sign(:, ones(1,D));
    
    % 6. Compute elite and best directions
    elite_dir = elite - popdecs;
    best_dir = best - elite;
    
    % 7. Combined mutation
    mutant = elite + ...
             cons_dir .* F(:, ones(1,D)) + ...
             elite_dir .* G(:, ones(1,D)) + ...
             best_dir .* H(:, ones(1,D));
    
    % 8. Adaptive crossover
    CR = 0.9 * (1 - (1:NP)'/NP).^0.2;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 9. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final clipping to ensure feasibility
    offspring = max(min(offspring, ub_rep), lb_rep);
end