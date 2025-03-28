% MATLAB Code
function [offspring] = updateFunc952(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select base vector considering both fitness and constraints
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_base = popdecs(feasible_mask, :);
        x_base = x_base(best_idx, :);
    else
        [~, least_violated] = min(cons);
        x_base = popdecs(least_violated, :);
    end
    
    % 2. Normalize constraints and fitness (0 to 1)
    c_min = min(cons);
    c_max = max(cons);
    c_norm = (cons - c_min) / (c_max - c_min + eps);
    
    f_min = min(popfits);
    f_max = max(popfits);
    f_norm = (popfits - f_min) / (f_max - f_min + eps);
    
    % 3. Compute adaptive scaling factors (0.3 to 0.9)
    F = 0.3 + 0.6 * (1 - c_norm) .* (1 - f_norm);
    
    % 4. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % 5. Enhanced directional mutation with fitness-weighted direction
    fitness_diff = popfits(r(:,3)) - popfits(r(:,4));
    fitness_sum = abs(popfits(r(:,3))) + abs(popfits(r(:,4))) + eps;
    w = fitness_diff ./ fitness_sum;
    
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    
    mutants = x_base(ones(NP,1),:) + ...
              F(:, ones(1,D)) .* diff1 + ...
              (w .* F)(:, ones(1,D)) .* diff2;
    
    % 6. Constraint-aware crossover rate (0.6 to 1.0)
    CR = 0.6 + 0.4 * (1 - c_norm);
    
    % 7. Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with reflection
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    % Handle lower bounds
    below_lb = offspring < lb_matrix;
    offspring(below_lb) = 2*lb_matrix(below_lb) - offspring(below_lb);
    
    % Handle upper bounds
    above_ub = offspring > ub_matrix;
    offspring(above_ub) = 2*ub_matrix(above_ub) - offspring(above_ub);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_matrix), ub_matrix);
end