% MATLAB Code
function [offspring] = updateFunc926(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Constraint-weighted fitness
    f_std = std(popfits);
    weighted_fits = popfits + 0.5 * max(0, cons) * f_std;
    
    % 2. Select best individual (feasible first)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(weighted_fits(feasible_mask));
        temp_indices = find(feasible_mask);
        x_best = popdecs(temp_indices(best_idx), :);
    else
        [~, best_idx] = min(weighted_fits);
        x_best = popdecs(best_idx, :);
    end
    
    % 3. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % 4. Adaptive parameters
    [~, rank_idx] = sort(weighted_fits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    
    % Normalized constraints
    cons_abs = abs(cons);
    cons_norm = cons_abs ./ (max(cons_abs) + eps);
    
    % Adaptive F and CR
    F = 0.5 + 0.3 * (ranks/NP);
    beta = 0.4 * (1 - cons_norm);
    CR = 0.7 - 0.3 * (ranks/NP);
    
    % 5. Enhanced mutation
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    x_best_rep = repmat(x_best, NP, 1);
    mutants = x_best_rep + F.*diff1 + beta.*diff2;
    
    % 6. Hybrid crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    gamma = 0.2;
    
    % Main crossover
    offspring(mask) = mutants(mask);
    
    % Directional update for non-crossover dimensions
    directional_update = ~mask & (rand(NP, D) < 0.2);
    offspring(directional_update) = popdecs(directional_update) + ...
        gamma .* (mutants(directional_update) - popdecs(directional_update));
    
    % 7. Bounce-back boundary handling
    below = offspring < lb;
    above = offspring > ub;
    offspring(below) = lb(below) + rand(sum(below(:)),1) .* (popdecs(below) - lb(below));
    offspring(above) = ub(above) - rand(sum(above(:)),1) .* (ub(above) - popdecs(above));
    
    % Final clamping
    offspring = max(min(offspring, ub), lb);
end