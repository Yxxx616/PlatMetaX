% MATLAB Code
function [offspring] = updateFunc1086(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Adaptive weight calculation
    mean_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    norm_fits = (popfits - mean_fit) / std_fit;
    sig_fits = 1./(1 + exp(-norm_fits));
    
    max_cons = max(abs(cons)) + eps;
    norm_cons = abs(cons) / max_cons;
    
    weights = 0.7*sig_fits + 0.3*(1 - norm_cons);
    weights = weights / (max(weights) + eps);
    
    % 2. Selection indices
    [~, best_idx] = min(popfits + 100*max(0, cons));
    elite_size = max(3, ceil(NP*0.2));
    [~, sorted_idx] = sort(popfits + 100*max(0, cons));
    elite_pool = sorted_idx(1:elite_size);
    
    idx_elite = elite_pool(randi(elite_size, NP, 1));
    idx_r1 = randi(NP, NP, 1);
    idx_r2 = randi(NP, NP, 1);
    idx_rand = randi(NP, NP, 1);
    
    % 3. Create vectors
    x_best = popdecs(best_idx(ones(NP,1)), :);
    x_elite = popdecs(idx_elite, :);
    x_rand = popdecs(idx_rand, :);
    x_r1 = popdecs(idx_r1, :);
    x_r2 = popdecs(idx_r2, :);
    
    % 4. Adaptive parameters
    F1 = min(max(0.5 + 0.1*randn(NP,1), 0.3), 0.7);
    F2 = 0.2 + 0.6*rand(NP,1);
    F1_mat = F1(:, ones(1,D));
    F2_mat = F2(:, ones(1,D));
    weights_mat = weights(:, ones(1,D));
    
    % 5. Mutation
    diff1 = x_r1 - x_r2;
    diff2 = x_best - x_rand;
    mutants = x_elite + F1_mat.*diff1 + F2_mat.*weights_mat.*diff2;
    
    % 6. Crossover with adaptive CR
    CR = 0.9 - 0.5*(1:NP)'/NP;
    CR_mat = CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR_mat;
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 7. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with bounce-back
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = lb_rep(below) + rand(NP,D).*(popdecs(below) - lb_rep(below));
    offspring(above) = ub_rep(above) - rand(NP,D).*(ub_rep(above) - popdecs(above));
    
    % Final bounds check
    offspring = min(max(offspring, lb_rep), ub_rep);
end