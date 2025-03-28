% MATLAB Code
function [offspring] = updateFunc1082(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Calculate weights combining fitness and constraints
    mean_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    norm_fits = (popfits - mean_fit) / std_fit;
    sig_fits = 1./(1 + exp(-5 * norm_fits));
    
    max_cons = max(abs(cons)) + eps;
    norm_cons = abs(cons) / max_cons;
    
    weights = sig_fits + 0.5 * norm_cons;
    weights = weights / (max(weights) + eps);
    
    % 2. Select indices
    [~, best_idx] = min(popfits + 100*max(0, cons));
    elite_size = ceil(NP*0.3);
    [~, sorted_idx] = sort(popfits + 100*max(0, cons));
    elite_pool = sorted_idx(1:elite_size);
    
    idx_best = best_idx(ones(NP,1));
    idx_elite = elite_pool(randi(elite_size, NP, 1));
    idx_r1 = randi(NP, NP, 1);
    idx_r2 = randi(NP, NP, 1);
    
    % 3. Calculate mutation vectors
    x_best = popdecs(idx_best, :);
    x_elite = popdecs(idx_elite, :);
    x_r1 = popdecs(idx_r1, :);
    x_r2 = popdecs(idx_r2, :);
    
    F = 0.5 * (1 + cos(pi * weights));
    F_mat = F(:, ones(1,D));
    
    base_term = (x_best - popdecs) + (x_r1 - x_r2);
    elite_term = (x_elite - popdecs) .* (weights(:, ones(1,D)) / max(weights);
    
    mutants = popdecs + F_mat .* base_term + elite_term;
    
    % 4. Crossover
    CR = 0.9 * (1 - tanh(weights));
    CR_mat = CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR_mat;
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 5. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 6. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Final bounds check
    offspring = min(max(offspring, lb_rep), ub_rep);
end