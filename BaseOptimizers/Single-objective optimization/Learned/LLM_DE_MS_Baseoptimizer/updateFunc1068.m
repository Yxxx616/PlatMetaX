% MATLAB Code
function [offspring] = updateFunc1068(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(abs(cons)) + eps;
    norm_cons = abs(cons) / c_max;
    
    % 2. Calculate weighted centroid and best individual
    weights = (1 - norm_fits) .* (1 - norm_cons);
    weights = weights / sum(weights);
    x_cent = weights' * popdecs;
    
    [~, best_idx] = min(popfits + 1000*norm_cons);
    x_best = popdecs(best_idx, :);
    
    % 3. Generate direction vectors
    delta_best = x_best(ones(NP,1), :) - popdecs;
    delta_cent = x_cent(ones(NP,1), :) - popdecs;
    
    % 4. Generate random differential vectors
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = idx(mod(1:NP, NP)+1);
    diff_vec = popdecs(r1,:) - popdecs(r2,:);
    
    % 5. Adaptive scaling factors
    F1 = 0.5 * (1 - norm_cons);
    F2 = 0.3 * (1 - norm_fits);
    F3 = 0.2 * (1 - norm_fits .* norm_cons);
    
    % 6. Create mutant vectors
    mutants = popdecs + ...
              F1(:, ones(1,D)) .* delta_best + ...
              F2(:, ones(1,D)) .* delta_cent + ...
              F3(:, ones(1,D)) .* diff_vec;
    
    % 7. Constraint-aware crossover
    CR = 0.9 * (1 - norm_cons).^2;
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