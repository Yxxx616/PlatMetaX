% MATLAB Code
function [offspring] = updateFunc940(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select base vector based on feasibility and fitness
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        temp_idx = find(feasible);
        x_base = popdecs(temp_idx(best_idx), :);
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
    F = 0.5 + 0.5 * c_norm .* (1 - f_norm);
    
    % 5. Enhanced directional mutation
    fitness_diff = popfits(r(:,3)) - popfits(r(:,4));
    fitness_denom = abs(popfits(r(:,3))) + abs(popfits(r(:,4))) + 1;
    direction_weights = fitness_diff ./ fitness_denom;
    
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    mutants = x_base(ones(NP,1),:) + ...
              F .* diff1 + ...
              (F .* direction_weights(:, ones(1,D))) .* diff2;
    
    % 6. Dynamic crossover rate
    CR = 0.9 * (1 - c_norm) + 0.1 * rand(NP, 1);
    
    % 7. Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Improved boundary handling
    out_of_bounds_low = offspring < lb;
    out_of_bounds_high = offspring > ub;
    
    offspring(out_of_bounds_low) = (popdecs(out_of_bounds_low) + lb(out_of_bounds_low)) / 2;
    offspring(out_of_bounds_high) = (popdecs(out_of_bounds_high) + ub(out_of_bounds_high)) / 2;
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end