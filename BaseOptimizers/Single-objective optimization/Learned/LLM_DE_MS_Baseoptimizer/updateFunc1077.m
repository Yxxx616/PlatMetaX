% MATLAB Code
function [offspring] = updateFunc1077(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(abs(cons)) + eps;
    norm_cons = abs(cons) / c_max;
    
    % 2. Calculate adaptive weights
    weights = (1 - norm_fits) .* (1 - norm_cons) + eps;
    weights = weights / sum(weights);
    
    % 3. Calculate reference points
    [~, best_idx] = min(popfits + 1000*norm_cons);
    x_best = popdecs(best_idx, :);
    x_cent = weights' * popdecs;
    
    [~, sorted_idx] = sort(popfits + 1000*norm_cons);
    elite_pool = sorted_idx(1:ceil(NP*0.3));
    elite_idx = elite_pool(randi(length(elite_pool), NP, 1));
    x_elite = popdecs(elite_idx, :);
    
    % 4. Generate random vectors for diversity
    idx = randperm(NP);
    x_rand = popdecs(idx, :);
    
    % 5. Calculate adaptive scaling factors
    F1 = 0.8 * (1 - norm_fits);
    F2 = 0.6 * (1 - norm_cons);
    F3 = 0.4 * rand(NP, 1);
    
    % 6. Create mutation vectors
    delta_best = x_best(ones(NP,1), :) - popdecs;
    delta_cent = x_cent(ones(NP,1), :) - popdecs;
    delta_elite = x_elite - x_rand;
    
    mutants = popdecs + F1(:, ones(1,D)) .* delta_best + ...
              F2(:, ones(1,D)) .* delta_cent + ...
              F3(:, ones(1,D)) .* delta_elite;
    
    % 7. Constraint-aware crossover
    CR = 0.9 * (1 - norm_cons.^2);
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:, ones(1,D));
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
    
    % 10. Final bounds check
    offspring = min(max(offspring, lb_rep), ub_rep);
end