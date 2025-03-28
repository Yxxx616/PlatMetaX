% MATLAB Code
function [offspring] = updateFunc1073(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(abs(cons)) + eps;
    norm_cons = abs(cons) / c_max;
    
    % 2. Calculate adaptive weights for centroid
    weights = (1 - norm_fits) .* (1 - norm_cons);
    weights = weights / sum(weights);
    x_cent = weights' * popdecs;
    
    % 3. Identify best individual considering constraints
    [~, best_idx] = min(popfits + 1000*norm_cons);
    x_best = popdecs(best_idx, :);
    
    % 4. Generate random indices
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = idx(mod(1:NP, NP)+1);
    r3 = idx(mod(2:NP+1, NP)+1);
    
    % 5. Calculate adaptive parameters
    F1 = 0.5 * (1 - norm_fits);
    F2 = 0.3 * (1 - norm_cons);
    F3 = 0.2 + 0.3 * norm_cons;
    w = 0.5 * (norm_fits + norm_cons);
    
    % 6. Create direction vectors
    delta_best = x_best(ones(NP,1), :) - popdecs;
    delta_cent = x_cent(ones(NP,1), :) - popdecs;
    delta_rand = popdecs(r1,:) - popdecs(r2,:) + popdecs(r3,:);
    
    % 7. Combine mutation strategies
    v = popdecs + F1(:, ones(1,D)) .* delta_best + F2(:, ones(1,D)) .* delta_cent;
    p = popdecs(r1,:) + F3(:, ones(1,D)) .* (popdecs(r2,:) - popdecs(r3,:));
    mutants = (1-w(:, ones(1,D))) .* v + w(:, ones(1,D)) .* p;
    
    % 8. Constraint-aware crossover
    CR = 0.9 * (1 - norm_cons.^2);
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:, ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 9. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 10. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % 11. Final bounds check
    offspring = min(max(offspring, lb_rep), ub_rep);
end