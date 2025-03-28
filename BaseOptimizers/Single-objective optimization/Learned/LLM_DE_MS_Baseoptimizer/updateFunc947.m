% MATLAB Code
function [offspring] = updateFunc947(popdecs, popfits, cons)
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
    
    % 3. Compute adaptive scaling factors (0.4 to 0.9)
    F = 0.4 + 0.5 * (1 - c_norm) .* (1 - f_norm);
    
    % 4. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % 5. Enhanced directional mutation
    fitness_ratio = popfits(r(:,3)) ./ (popfits(r(:,4)) + eps);
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    
    mutants = x_base(ones(NP,1),:) + ...
              F(:, ones(1,D)) .* diff1 + ...
              0.7 * F(:, ones(1,D)) .* (1 - fitness_ratio(:, ones(1,D))) .* diff2;
    
    % 6. Adaptive crossover rate (0.05 to 0.9)
    CR = 0.9 * (1 - c_norm) + 0.05;
    
    % 7. Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Improved boundary handling with base vector reference
    out_of_bounds = offspring < lb | offspring > ub;
    rand_offsets = rand(sum(out_of_bounds(:)), 1) .* (ub(out_of_bounds) - lb(out_of_bounds));
    offspring(out_of_bounds) = x_base(out_of_bounds) + rand_offsets;
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end