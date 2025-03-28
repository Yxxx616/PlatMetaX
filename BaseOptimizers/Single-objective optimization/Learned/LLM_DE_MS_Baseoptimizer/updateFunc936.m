% MATLAB Code
function [offspring] = updateFunc936(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify best individual considering constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        temp_idx = find(feasible);
        x_best = popdecs(temp_idx(best_idx), :);
        f_avg = mean(popfits(feasible));
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx, :);
        f_avg = mean(popfits);
    end
    
    % 2. Normalize constraint violations
    c_norm = abs(cons) / (max(abs(cons)) + eps);
    
    % 3. Rank population based on fitness
    [~, fit_rank] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(fit_rank) = (1:NP)';
    
    % 4. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % 5. Adaptive parameters
    F = 0.5 * (1 + ranks/NP) .* (1 - c_norm);
    CR = 0.9 * (1 - ranks/NP);
    
    % 6. Enhanced mutation strategy
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    fitness_ratio = f_avg./(abs(popfits) + eps);
    mutants = x_best(ones(NP,1),:) + F.*diff1 + F.*fitness_ratio.*diff2;
    
    % 7. Constraint-aware crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Improved boundary handling
    below = offspring < lb;
    above = offspring > ub;
    offspring(below) = (popdecs(below) + lb(below)) / 2;
    offspring(above) = (popdecs(above) + ub(above)) / 2;
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end