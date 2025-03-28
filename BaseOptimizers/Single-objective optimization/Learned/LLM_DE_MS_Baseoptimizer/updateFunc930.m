% MATLAB Code
function [offspring] = updateFunc930(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Find best individual considering constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        temp_idx = find(feasible);
        x_best = popdecs(temp_idx(best_idx), :);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx, :);
    end
    
    % 2. Rank population based on combined fitness and constraints
    [~, fit_rank] = sort(popfits);
    [~, cons_rank] = sort(abs(cons));
    combined_rank = 0.7*fit_rank + 0.3*cons_rank;
    [~, rank_order] = sort(combined_rank);
    ranks = zeros(NP, 1);
    ranks(rank_order) = 1:NP;
    
    % 3. Normalize constraint violations
    c_max = max(abs(cons)) + eps;
    c_norm = abs(cons) / c_max;
    
    % 4. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 5);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 5));
    end
    
    % 5. Adaptive scaling factors
    F1 = 0.5 * (1 + ranks/NP);
    F2 = 0.3 * (1 - ranks/NP);
    F3 = 0.2 * c_norm;
    
    % 6. Enhanced mutation with three components
    term1 = popdecs(r(:,1),:) - popdecs(r(:,1),:);
    term2 = x_best - popdecs(r(:,1),:);
    term3 = popdecs(r(:,2),:) - popdecs(r(:,3),:);
    term4 = popdecs(r(:,4),:) - popdecs(r(:,5),:);
    
    mutants = popdecs(r(:,1),:) + F1.*term2 + F2.*term3 + ...
              F3.*term4;
    
    % 7. Crossover with adaptive CR
    CR = 0.5 + 0.4*(1 - ranks/NP);
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