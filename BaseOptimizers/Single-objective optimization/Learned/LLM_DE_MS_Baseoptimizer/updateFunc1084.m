% MATLAB Code
function [offspring] = updateFunc1084(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute weights combining fitness and constraints
    mean_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    norm_fits = (popfits - mean_fit) / std_fit;
    sig_fits = 1./(1 + exp(-norm_fits));  % Sigmoid normalization
    
    max_cons = max(abs(cons)) + eps;
    norm_cons = abs(cons) / max_cons;
    
    weights = 0.7*sig_fits + 0.3*norm_cons;  % Weighted combination
    weights = weights / (max(weights) + eps);
    
    % 2. Select indices
    [~, best_idx] = min(popfits + 100*max(0, cons));  % Best considering constraints
    elite_size = max(2, ceil(NP*0.2));
    [~, sorted_idx] = sort(popfits + 100*max(0, cons));
    elite_pool = sorted_idx(1:elite_size);
    
    idx_best = best_idx(ones(NP,1));
    idx_elite = elite_pool(randi(elite_size, NP, 1));
    idx_r1 = randi(NP, NP, 1);
    idx_r2 = randi(NP, NP, 1);
    idx_r3 = randi(NP, NP, 1);
    idx_r4 = randi(NP, NP, 1);
    idx_rand = randi(NP, NP, 1);
    
    % 3. Create vectors
    x_best = popdecs(idx_best, :);
    x_elite = popdecs(idx_elite, :);
    x_rand = popdecs(idx_rand, :);
    x_r1 = popdecs(idx_r1, :);
    x_r2 = popdecs(idx_r2, :);
    x_r3 = popdecs(idx_r3, :);
    x_r4 = popdecs(idx_r4, :);
    
    % 4. Adaptive parameters
    F = 0.5 + 0.3 * (1 - weights);  % Higher F for worse solutions
    F_mat = F(:, ones(1,D));
    CR = 0.1 + 0.8 * weights;  % Higher CR for better solutions
    CR_mat = CR(:, ones(1,D));
    
    % 5. Mutation strategy
    elite_term = x_elite + F_mat .* (x_r1 - x_r2);
    constraint_term = 0.5 * (1 + norm_cons(:, ones(1,D))) .* (x_best - x_rand);
    diversity_term = 0.3 * (1 - weights(:, ones(1,D))) .* (x_r3 - x_r4);
    
    mutants = elite_term + constraint_term + diversity_term;
    
    % 6. Crossover
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
    
    offspring(below) = lb_rep(below) + rand(NP,D) .* (popdecs(below) - lb_rep(below));
    offspring(above) = ub_rep(above) - rand(NP,D) .* (ub_rep(above) - popdecs(above));
    
    % Final bounds check
    offspring = min(max(offspring, lb_rep), ub_rep);
end