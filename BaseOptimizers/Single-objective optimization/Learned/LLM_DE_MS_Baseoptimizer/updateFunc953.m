% MATLAB Code
function [offspring] = updateFunc953(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select base vector with adaptive blending
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = popdecs(feasible_mask, :);
        x_best = x_best(best_idx, :);
    else
        x_best = mean(popdecs, 1); % fallback centroid
    end
    
    [~, least_violated] = min(cons);
    x_lv = popdecs(least_violated, :);
    
    % Adaptive blending factor for infeasible cases
    alpha = min(cons) / (min(cons) - median(cons) + eps);
    alpha = max(0, min(1, alpha));
    x_base = x_lv + alpha * (x_best - x_lv);
    
    % 2. Normalize constraints and fitness
    c_min = min(cons); c_max = max(cons);
    c_norm = (cons - c_min) / (c_max - c_min + eps);
    
    f_min = min(popfits); f_max = max(popfits);
    f_norm = (popfits - f_min) / (f_max - f_min + eps);
    
    % 3. Compute adaptive parameters
    F = 0.4 + 0.5 * (1 - c_norm) .* (1 - f_norm);
    CR = 0.7 + 0.3 * (1 - c_norm);
    
    % 4. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % 5. Enhanced directional mutation
    fitness_diff = popfits(r(:,3)) - popfits(r(:,4));
    fitness_sum = abs(popfits(r(:,3))) + abs(popfits(r(:,4))) + eps;
    lambda = fitness_diff ./ fitness_sum;
    
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    
    mutants = x_base(ones(NP,1),:) + ...
              F(:, ones(1,D)) .* diff1 + ...
              (lambda .* F)(:, ones(1,D)) .* diff2;
    
    % 6. Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Improved boundary handling
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    % Random reflection factor
    beta = rand(NP, D);
    
    % Handle lower bounds
    below_lb = offspring < lb_matrix;
    offspring(below_lb) = lb_matrix(below_lb) + beta(below_lb) .* ...
                         (lb_matrix(below_lb) - offspring(below_lb));
    
    % Handle upper bounds
    above_ub = offspring > ub_matrix;
    offspring(above_ub) = ub_matrix(above_ub) - beta(above_ub) .* ...
                         (offspring(above_ub) - ub_matrix(above_ub));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_matrix), ub_matrix);
end