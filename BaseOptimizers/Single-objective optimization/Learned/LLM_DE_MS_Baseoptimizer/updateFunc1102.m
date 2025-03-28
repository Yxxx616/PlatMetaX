% MATLAB Code
function [offspring] = updateFunc1102(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Calculate adaptive weights (60% fitness, 40% constraints)
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons)) + eps;
    
    w_f = (popfits - f_min) / (f_max - f_min + eps);
    w_c = 1 - abs(cons) / (c_max + eps);
    weights = 0.6 * w_f + 0.4 * w_c;
    
    % 2. Create three-tier pools with dynamic sizes
    [~, sorted_idx] = sort(weights);
    elite_size = max(2, floor(NP*0.25));
    diverse_size = max(2, floor(NP*0.15));
    
    elite_pool = sorted_idx(end-elite_size+1:end);
    medium_pool = sorted_idx(elite_size+1:end-diverse_size);
    diverse_pool = sorted_idx(1:diverse_size);
    
    % 3. Generate indices for mutation
    idx_e1 = elite_pool(randi(elite_size, NP, 1));
    idx_e2 = elite_pool(randi(elite_size, NP, 1));
    idx_m = medium_pool(randi(length(medium_pool), NP, 1));
    idx_d = diverse_pool(randi(diverse_size, NP, 1));
    
    % 4. Adaptive mutation factors
    F1 = 0.5 * weights + 0.3;
    F2 = 0.8 - 0.5 * weights;
    
    x_e1 = popdecs(idx_e1, :);
    x_e2 = popdecs(idx_e2, :);
    x_m = popdecs(idx_m, :);
    x_d = popdecs(idx_d, :);
    
    mutants = popdecs + F1.*(x_e1 - x_e2) + F2.*(x_m - x_d);
    
    % 5. Directional crossover with adaptive rate
    CR = 0.9 - 0.6 * weights;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 6. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Enhanced boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflective projection with clamping
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = min(2*lb_rep(below) - offspring(below), ub_rep(below));
    offspring(above) = max(2*ub_rep(above) - offspring(above), lb_rep(above));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end