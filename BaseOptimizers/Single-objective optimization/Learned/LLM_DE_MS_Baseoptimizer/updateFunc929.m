% MATLAB Code
function [offspring] = updateFunc929(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    f_norm = (popfits - f_min) ./ f_range;
    
    c_abs = abs(cons);
    c_min = min(c_abs);
    c_max = max(c_abs);
    c_range = c_max - c_min + eps;
    c_norm = (c_abs - c_min) ./ c_range;
    
    % 2. Calculate selection probabilities
    p = 1 ./ (1 + f_norm + 0.5*c_norm);
    [~, rank_idx] = sort(p, 'descend');
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    
    % 3. Find best individual (considering constraints)
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        temp_idx = find(feasible);
        x_best = popdecs(temp_idx(best_idx), :);
    else
        [~, best_idx] = min(f_norm + 0.5*c_norm);
        x_best = popdecs(best_idx, :);
    end
    
    % 4. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 5);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 5));
    end
    
    % 5. Adaptive parameters
    F = 0.4 + 0.4 * (ranks/NP);
    CR = 0.1 + 0.8 * (1 - ranks/NP);
    
    % 6. Enhanced mutation
    term1 = popdecs(r(:,1),:) - popdecs(r(:,1),:);
    term2 = x_best - popdecs(r(:,1),:);
    term3 = popdecs(r(:,2),:) - popdecs(r(:,3),:);
    term4 = popdecs(r(:,4),:) - popdecs(r(:,5),:);
    
    mutants = popdecs(r(:,1),:) + F.*term2 + F.*term3 + ...
              0.5*(1-f_norm).*term4;
    
    % 7. Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with midpoint reflection
    below = offspring < lb;
    above = offspring > ub;
    offspring(below) = (popdecs(below) + lb(below)) / 2;
    offspring(above) = (popdecs(above) + ub(above)) / 2;
    
    % Final clamping
    offspring = max(min(offspring, ub), lb);
end