% MATLAB Code
function [offspring] = updateFunc1091(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Calculate weights
    max_cons = max(abs(cons)) + eps;
    w_c = 1 - abs(cons)/max_cons;
    
    [~, ranks] = sort(popfits);
    w_f = ranks'/NP;
    
    weights = 0.6*w_c + 0.4*w_f;
    
    % 2. Select elite pool (top 40%)
    elite_size = max(3, ceil(NP*0.4));
    [~, sorted_idx] = sort(weights, 'descend');
    elite_pool = sorted_idx(1:elite_size);
    
    % 3. Generate indices for mutation
    idx_elite1 = elite_pool(randi(elite_size, NP, 1));
    idx_elite2 = elite_pool(randi(elite_size, NP, 1));
    idx_r1 = randi(NP, NP, 1);
    idx_r2 = randi(NP, NP, 1);
    
    % 4. Create direction vectors
    x_elite1 = popdecs(idx_elite1, :);
    x_elite2 = popdecs(idx_elite2, :);
    x_r1 = popdecs(idx_r1, :);
    x_r2 = popdecs(idx_r2, :);
    
    % 5. Adaptive scaling factors
    F_e = 0.5 + 0.3*weights;
    F_r = 0.2*(1 - weights);
    
    % 6. Mutation
    diff_elite = x_elite1 - x_elite2;
    diff_rand = x_r1 - x_r2;
    
    mutants = popdecs + F_e.*diff_elite + F_r.*diff_rand;
    
    % 7. Crossover
    CR = 0.85 - 0.35*weights;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 8. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Final bounds check and repair if still out of bounds
    offspring = min(max(offspring, lb_rep), ub_rep);
end