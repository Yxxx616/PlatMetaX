% MATLAB Code
function [offspring] = updateFunc922(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Constraint-aware best selection
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible,:);
        x_best = x_best(best_idx,:);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx,:);
    end
    
    % 2. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 6);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 6));
    end
    
    % 3. Adaptive parameters
    cv_max = max(abs(cons)) + eps;
    F = 0.3 + 0.5 * (1 - abs(cons)/cv_max);
    
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.5 + 0.3 * (ranks/NP);
    
    % 4. Enhanced directional mutation
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    diff3 = popdecs(r(:,5),:) - popdecs(r(:,6),:);
    x_best_rep = repmat(x_best, NP, 1);
    
    % Dynamic weighting for exploration
    w = 0.8; % Fixed weight for second difference
    mutants = x_best_rep + F.*diff1 + w*diff2 + 0.2*diff3;
    
    % 5. Crossover with jrand enforcement
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % 6. Create offspring with dynamic boundary handling
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with dynamic reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = (lb_rep(below) + popdecs(below))/2;
    offspring(above) = (ub_rep(above) + popdecs(above))/2;
    
    % Final clamping
    offspring = max(min(offspring, ub_rep), lb_rep);
end