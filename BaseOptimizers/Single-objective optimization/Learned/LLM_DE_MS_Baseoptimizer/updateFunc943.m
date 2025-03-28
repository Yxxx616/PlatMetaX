% MATLAB Code
function [offspring] = updateFunc943(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select base vector (best feasible or least violated)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_base = popdecs(feasible_mask, :);
        x_base = x_base(best_idx, :);
    else
        [~, least_violated] = min(cons);
        x_base = popdecs(least_violated, :);
    end
    
    % 2. Normalize constraints and fitness
    c_min = min(cons);
    c_max = max(cons);
    c_norm = (cons - c_min) / (c_max - c_min + eps);
    
    f_min = min(popfits);
    f_max = max(popfits);
    f_norm = (popfits - f_min) / (f_max - f_min + eps);
    
    % 3. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % 4. Compute adaptive scaling factors
    F = 0.5 * (1 + c_norm) .* (1 - f_norm);
    
    % 5. Directional mutation with fitness guidance
    fitness_diff = popfits(r(:,3)) - popfits(r(:,4));
    fitness_denom = abs(popfits(r(:,3))) + abs(popfits(r(:,4))) + 1;
    alpha = 0.5 * F .* (fitness_diff ./ fitness_denom);
    
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    
    % Vectorized mutation
    mutants = x_base(ones(NP,1),:) + ...
              F(:, ones(1,D)) .* diff1 + ...
              alpha(:, ones(1,D)) .* diff2;
    
    % 6. Dynamic crossover rate
    CR = 0.9 * (1 - c_norm) + 0.1 * rand(NP, 1);
    
    % 7. Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Bounce-back boundary handling with base vector reference
    below_lb = offspring < lb;
    above_ub = offspring > ub;
    
    offspring(below_lb) = lb(below_lb) + rand(sum(below_lb(:)), 1) .* ...
                         (x_base(below_lb) - lb(below_lb));
    offspring(above_ub) = ub(above_ub) - rand(sum(above_ub(:)), 1) .* ...
                         (x_base(above_ub) - ub(above_ub));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end