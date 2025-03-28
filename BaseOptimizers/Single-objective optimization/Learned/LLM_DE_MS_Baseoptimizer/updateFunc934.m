% MATLAB Code
function [offspring] = updateFunc934(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select best individual considering constraints
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
    [~, cons_rank] = sort(cons);
    combined_rank = 0.7*fit_rank + 0.3*cons_rank;
    [~, rank_order] = sort(combined_rank);
    ranks = 1:NP;
    ranks(rank_order) = ranks;
    
    % 3. Normalize constraint violations and ranks
    c_norm = abs(cons) / (max(abs(cons)) + eps);
    r_norm = (ranks' - 1) / (NP - 1);
    
    % 4. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % 5. Adaptive parameters
    F = 0.5 + 0.3*(1 - c_norm) + 0.2*rand(NP,1);
    CR = 0.9 - 0.3*r_norm;
    
    % 6. Novel mutation strategy
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    mutants = x_best(ones(NP,1),:) + F.*diff1 + F.*diff2.*(1 - c_norm);
    
    % 7. Constraint-aware crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Adaptive boundary handling
    below = offspring < lb;
    above = offspring > ub;
    offspring(below) = (x_best(below) + lb(below)) / 2;
    offspring(above) = (x_best(above) + ub(above)) / 2;
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end