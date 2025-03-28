% MATLAB Code
function [offspring] = updateFunc925(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Constraint-weighted fitness calculation
    f_std = std(popfits);
    weighted_fits = popfits + 0.5 * max(0, cons) * f_std;
    
    % 2. Select best individual considering constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(weighted_fits(feasible));
        x_best = popdecs(feasible(best_idx),:);
    else
        [~, best_idx] = min(weighted_fits);
        x_best = popdecs(best_idx,:);
    end
    
    % 3. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % 4. Adaptive parameters
    cons_norm = (cons - min(cons)) ./ (max(cons) - min(cons) + eps);
    [~, rank_idx] = sort(weighted_fits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    
    F = 0.4 + 0.4 * (ranks/NP);
    CR = 0.5 + 0.3 * cons_norm;
    
    % 5. Enhanced mutation with constraint adaptation
    beta = 0.3 * (1 - abs(cons_norm));
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    x_best_rep = repmat(x_best, NP, 1);
    
    mutants = x_best_rep + F.*diff1 + beta.*diff2;
    
    % 6. Hybrid crossover with directional update
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    gamma = 0.2;
    
    % Main crossover
    offspring(mask) = mutants(mask);
    
    % Directional update for non-crossover dimensions
    directional_update = ~mask & (rand(NP, D) < 0.3);
    offspring(directional_update) = popdecs(directional_update) + ...
        gamma .* (mutants(directional_update) - popdecs(directional_update));
    
    % 7. Reflective boundary handling
    below = offspring < lb;
    above = offspring > ub;
    offspring(below) = 2*lb(below) - offspring(below);
    offspring(above) = 2*ub(above) - offspring(above);
    
    % Final clamping
    offspring = max(min(offspring, ub), lb);
end