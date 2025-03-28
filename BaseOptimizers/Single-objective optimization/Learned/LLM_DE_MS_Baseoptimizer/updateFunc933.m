% MATLAB Code
function [offspring] = updateFunc933(popdecs, popfits, cons)
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
    
    % 2. Rank population based on fitness and constraints
    [~, fit_rank] = sort(popfits);
    [~, cons_rank] = sort(abs(cons));
    combined_rank = 0.6*fit_rank + 0.4*cons_rank;
    [~, rank_order] = sort(combined_rank);
    ranks = 1:NP;
    ranks(rank_order) = ranks;
    
    % 3. Normalize constraint violations and ranks
    c_norm = abs(cons) / (max(abs(cons)) + eps);
    r_norm = (ranks' - 1) / (NP - 1);
    
    % 4. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 7);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 7));
    end
    
    % 5. Adaptive parameters
    F1 = 0.8 + 0.1 * (1 - r_norm);
    F2 = 0.6 + 0.2 * (1 - c_norm);
    F3 = 0.4 + 0.3 * rand(NP,1);
    
    w1 = 0.5 * (1 - c_norm);
    w2 = 0.3 + 0.2 * (1 - r_norm);
    w3 = 1 - w1 - w2;
    
    % 6. Three-component mutation (vectorized)
    v1 = x_best(ones(NP,1),:) + F1.*(popdecs(r(:,1),:) - popdecs(r(:,2),:)).*(1 - c_norm);
    v2 = popdecs + F2.*(popdecs(r(:,3),:) - popdecs(r(:,4),:)).*(1 - r_norm);
    v3 = popdecs(r(:,5),:) + F3.*(popdecs(r(:,6),:) - popdecs(r(:,7),:)).*c_norm;
    
    mutants = w1.*v1 + w2.*v2 + w3.*v3;
    
    % 7. Adaptive crossover
    CR = 0.7 + 0.2*(1 - r_norm);
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
    
    % Ensure bounds are respected
    offspring = min(max(offspring, lb), ub);
end