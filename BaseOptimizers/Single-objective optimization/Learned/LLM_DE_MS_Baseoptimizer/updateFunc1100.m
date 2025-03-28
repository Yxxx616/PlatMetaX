% MATLAB Code
function [offspring] = updateFunc1100(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Calculate composite weights (60% fitness, 40% constraints)
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons)) + eps;
    
    w_f = (popfits - f_min) / (f_max - f_min + eps);
    w_c = 1 - abs(cons) / (c_max + eps);
    weights = 0.6 * w_f + 0.4 * w_c;
    
    % 2. Create elite and diverse pools (25% population each)
    pool_size = max(3, floor(NP*0.25));
    [~, sorted_idx] = sort(weights);
    elite_pool = sorted_idx(end-pool_size+1:end);
    diverse_pool = sorted_idx(1:pool_size);
    
    % 3. Generate indices for mutation
    idx_e1 = elite_pool(randi(pool_size, NP, 1));
    idx_e2 = elite_pool(randi(pool_size, NP, 1));
    idx_d1 = diverse_pool(randi(pool_size, NP, 1));
    idx_d2 = diverse_pool(randi(pool_size, NP, 1));
    
    % 4. Adaptive mutation with directional guidance
    F = 0.2 + 0.6 * weights;
    x_e1 = popdecs(idx_e1, :);
    x_e2 = popdecs(idx_e2, :);
    x_d1 = popdecs(idx_d1, :);
    x_d2 = popdecs(idx_d2, :);
    
    mutants = popdecs + F.*(x_e1 - x_e2) + (1-F).*(x_d1 - x_d2);
    
    % 5. Directional crossover with adaptive rate
    CR = 0.85 - 0.4 * weights;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 6. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Reflective boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end