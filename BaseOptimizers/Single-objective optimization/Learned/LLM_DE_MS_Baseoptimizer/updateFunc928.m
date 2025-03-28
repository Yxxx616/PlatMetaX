% MATLAB Code
function [offspring] = updateFunc928(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize constraints and fitness
    cons_abs = abs(cons);
    c_min = min(cons_abs);
    c_max = max(cons_abs);
    c_range = c_max - c_min + eps;
    c_norm = (cons_abs - c_min) ./ c_range;
    
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    f_norm = (popfits - f_min) ./ f_range;
    
    % 2. Combined fitness (lower is better)
    combined_fits = f_norm + 0.5 * c_norm;
    [~, rank_idx] = sort(combined_fits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    
    % 3. Select best individual (considering constraints)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        temp_indices = find(feasible_mask);
        x_best = popdecs(temp_indices(best_idx), :);
    else
        [~, best_idx] = min(combined_fits);
        x_best = popdecs(best_idx, :);
    end
    
    % 4. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % 5. Adaptive parameters
    F = 0.5 * (1 + c_norm);  % More exploration for constrained individuals
    beta = 0.3 * (1 - f_norm); % More exploitation for better fitness
    CR = 0.9 * (1 - ranks/NP); % Higher CR for better individuals
    
    % 6. Enhanced mutation
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = x_best - popdecs(r(:,3),:);
    mutants = popdecs(r(:,4),:) + F.*diff1 + beta.*diff2;
    
    % 7. Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with reflection
    below = offspring < lb;
    above = offspring > ub;
    offspring(below) = 2*lb(below) - offspring(below);
    offspring(above) = 2*ub(above) - offspring(above);
    
    % Final clamping
    offspring = max(min(offspring, ub), lb);
end