% MATLAB Code
function [offspring] = updateFunc1087(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize constraints and calculate violation weights
    max_cons = max(abs(cons)) + eps;
    norm_cons = abs(cons) / max_cons;
    cons_weights = 1 - norm_cons;
    
    % 2. Rank-based fitness weights
    [~, ranks] = sort(popfits);
    rank_weights = (ranks/NP)';
    
    % 3. Select indices
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
    else
        [~, best_idx] = min(popfits + 100*max(0, cons));
    end
    
    elite_size = max(3, ceil(NP*0.2));
    [~, sorted_idx] = sort(popfits + 100*max(0, cons));
    elite_pool = sorted_idx(1:elite_size);
    
    idx_elite = elite_pool(randi(elite_size, NP, 1));
    idx_r1 = randi(NP, NP, 1);
    idx_r2 = randi(NP, NP, 1);
    idx_rand = randi(NP, NP, 1);
    
    % 4. Create base vectors
    x_best = popdecs(best_idx(ones(NP,1)), :);
    x_elite = popdecs(idx_elite, :);
    x_rand = popdecs(idx_rand, :);
    x_r1 = popdecs(idx_r1, :);
    x_r2 = popdecs(idx_r2, :);
    
    % 5. Adaptive parameters
    F_base = 0.5 * (1 + rank_weights);
    alpha = 0.3 + 0.5 * rand(NP,1);
    F = F_base .* (0.8 + 0.4 * rand(NP,1));
    
    % 6. Constraint-aware direction
    cons_dir = sign(cons) .* norm_cons;
    cons_dir_mat = cons_dir(:, ones(1,D));
    
    % 7. Mutation
    diff1 = x_r1 - x_r2;
    diff2 = x_best - x_rand;
    mutants = x_elite + ...
              F(:, ones(1,D)) .* diff2 + ...
              (1 - alpha(:, ones(1,D))) .* diff1 + ...
              alpha(:, ones(1,D)) .* cons_dir_mat;
    
    % 8. Crossover
    CR = 0.9 - 0.4 * rank_weights;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 9. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 10. Boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = lb_rep(below) + rand(NP,D).*(popdecs(below) - lb_rep(below));
    offspring(above) = ub_rep(above) - rand(NP,D).*(ub_rep(above) - popdecs(above));
    
    % Final bounds check
    offspring = min(max(offspring, lb_rep), ub_rep);
end