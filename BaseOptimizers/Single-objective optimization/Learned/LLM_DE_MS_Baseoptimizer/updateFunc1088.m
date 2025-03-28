% MATLAB Code
function [offspring] = updateFunc1088(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Calculate constraint weights
    max_cons = max(abs(cons)) + eps;
    w_c = 1 - abs(cons)/max_cons;
    
    % 2. Calculate fitness weights (rank-based)
    [~, ranks] = sort(popfits);
    w_f = ranks'/NP;
    
    % 3. Composite weights (emphasize constraints)
    alpha = 0.7;
    weights = alpha*w_c + (1-alpha)*w_f;
    
    % 4. Select elite pool (top 30%)
    elite_size = max(3, ceil(NP*0.3));
    [~, sorted_idx] = sort(weights, 'descend');
    elite_pool = sorted_idx(1:elite_size);
    
    % 5. Generate indices for mutation
    idx_elite1 = elite_pool(randi(elite_size, NP, 1));
    idx_elite2 = elite_pool(randi(elite_size, NP, 1));
    idx_r1 = randi(NP, NP, 1);
    idx_r2 = randi(NP, NP, 1);
    
    % 6. Create direction vectors
    x_elite1 = popdecs(idx_elite1, :);
    x_elite2 = popdecs(idx_elite2, :);
    x_r1 = popdecs(idx_r1, :);
    x_r2 = popdecs(idx_r2, :);
    
    % 7. Adaptive scaling factors
    F1 = 0.8 * weights(:, ones(1,D));
    F2 = 0.5 * (1 - weights(:, ones(1,D)));
    F3 = 0.3 * sign(cons(:, ones(1,D))) .* w_c(:, ones(1,D));
    
    % 8. Mutation
    diff_elite = x_elite1 - x_elite2;
    diff_rand = x_r1 - x_r2;
    mutants = popdecs + F1.*diff_elite + F2.*diff_rand + F3;
    
    % 9. Crossover
    CR = 0.9 - 0.5*weights;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 10. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 11. Boundary handling with repair
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = (popdecs(below) + lb_rep(below))/2 + ...
        rand(NP,D).*(popdecs(below) - lb_rep(below))/2;
    offspring(above) = (popdecs(above) + ub_rep(above))/2 - ...
        rand(NP,D).*(ub_rep(above) - popdecs(above))/2;
    
    % Final bounds check
    offspring = min(max(offspring, lb_rep), ub_rep);
end