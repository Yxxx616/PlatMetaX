% MATLAB Code
function [offspring] = updateFunc959(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute weighted centroid
    f_mean = mean(popfits);
    f_std = std(popfits) + eps;
    c_max = max(abs(cons)) + eps;
    
    weights = 1./(1 + max(0, cons) + abs(popfits - f_mean)/f_std);
    weights = weights / sum(weights);
    centroid = weights' * popdecs;
    
    % 2. Identify best solution
    [~, best_idx] = min(popfits + 10*max(0, cons));
    x_best = popdecs(best_idx,:);
    
    % 3. Compute opposition points
    opposition = lb + ub - popdecs;
    
    % 4. Adaptive scaling factors
    F1 = 0.5 * (1 + abs(cons)/c_max);
    F2 = 0.3 * (1 - abs(cons)/c_max);
    
    % 5. Enhanced mutation with opposition learning
    mutants = centroid(ones(NP,1),:) + ...
              F1.*(x_best(ones(NP,1),:) - popdecs) + ...
              F2.*(opposition - popdecs);
    
    % 6. Adaptive CR based on constraints
    CR = 0.7 * (1 - abs(cons)/c_max).^0.5;
    
    % 7. Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with random reinitialization
    lb_matrix = repmat(lb, NP, 1);
    ub_matrix = repmat(ub, NP, 1);
    
    out_of_bounds = offspring < lb_matrix | offspring > ub_matrix;
    rand_reinit = lb_matrix + rand(NP, D).*(ub_matrix - lb_matrix);
    offspring(out_of_bounds) = rand_reinit(out_of_bounds);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_matrix), ub_matrix);
end