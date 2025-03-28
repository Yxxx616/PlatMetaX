% MATLAB Code
function [offspring] = updateFunc1083(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize fitness and constraints for weights
    mean_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    norm_fits = (popfits - mean_fit) / std_fit;
    sig_fits = 1./(1 + exp(-5 * norm_fits));
    
    max_cons = max(abs(cons)) + eps;
    norm_cons = abs(cons) / max_cons;
    
    weights = sig_fits + norm_cons;
    weights = weights / (max(weights) + eps);
    
    % 2. Select indices
    [~, best_idx] = min(popfits + 100*max(0, cons));
    elite_size = ceil(NP*0.2);
    [~, sorted_idx] = sort(popfits + 100*max(0, cons));
    elite_pool = sorted_idx(1:elite_size);
    
    idx_best = best_idx(ones(NP,1));
    idx_elite = elite_pool(randi(elite_size, NP, 1));
    idx_r1 = randi(NP, NP, 1);
    idx_r2 = randi(NP, NP, 1);
    idx_r3 = randi(NP, NP, 1);
    idx_rand = randi(NP, NP, 1);
    
    % 3. Create base vectors
    x_best = popdecs(idx_best, :);
    x_elite = popdecs(idx_elite, :);
    x_rand = popdecs(idx_rand, :);
    x_r1 = popdecs(idx_r1, :);
    x_r2 = popdecs(idx_r2, :);
    x_r3 = popdecs(idx_r3, :);
    
    % 4. Adaptive parameters
    F = 0.5 + 0.3 * cos(pi * weights);
    F_mat = F(:, ones(1,D));
    CR = 0.9 * (1 - weights);
    CR_mat = CR(:, ones(1,D));
    
    % 5. Mutation strategy
    base_term = x_best + 0.5*(x_elite - x_rand);
    diff_term = (x_r1 - x_r2);
    explore_term = 0.2 * norm_cons(:, ones(1,D)) .* (x_rand - x_r3);
    
    mutants = base_term + F_mat .* diff_term + explore_term;
    
    % 6. Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR_mat;
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 7. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Final bounds check
    offspring = min(max(offspring, lb_rep), ub_rep);
end