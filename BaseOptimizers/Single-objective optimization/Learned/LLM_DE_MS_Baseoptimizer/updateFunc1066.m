% MATLAB Code
function [offspring] = updateFunc1066(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(abs(cons)) + eps;
    norm_cons = abs(cons) / c_max;
    
    % 2. Calculate weighted centroid
    weights = (1 - norm_fits) + norm_cons;
    weights = weights / sum(weights);
    x_cent = weights' * popdecs;
    
    % 3. Identify best and worst individuals
    [~, best_idx] = min(popfits + 1000*norm_cons);
    [~, worst_idx] = max(popfits + 1000*norm_cons);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    % 4. Adaptive scaling factors
    F = 0.4 + 0.4 * norm_cons;
    F = F(:, ones(1, D));
    
    % 5. Generate random indices
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = idx(mod(1:NP, NP)+1);
    r3 = idx(mod(2:NP+1, NP)+1);
    r4 = idx(mod(3:NP+2, NP)+1);
    
    % 6. Create mutant vectors
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    best_diff = x_best(ones(NP,1), :) - x_worst(ones(NP,1), :);
    
    mutants = x_cent(ones(NP,1), :) + ...
              F .* (diff1 + diff2) + ...
              0.1 * best_diff;
    
    % 7. Constraint-aware crossover
    CR = 0.9 - 0.5 * norm_cons;
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