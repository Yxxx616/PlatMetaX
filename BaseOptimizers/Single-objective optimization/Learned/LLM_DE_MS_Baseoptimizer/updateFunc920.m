% MATLAB Code
function [offspring] = updateFunc920(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select best individual considering constraints
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        feasible_indices = find(feasible_mask);
        x_best = popdecs(feasible_indices(best_idx), :);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx, :);
    end
    
    % 2. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i, :) = candidates(randperm(length(candidates), 4));
    end
    
    % 3. Compute adaptive F based on constraints
    cv_max = max(abs(cons)) + eps;
    F = 0.5 + 0.3 * tanh(1 - abs(cons)/cv_max);
    
    % 4. Rank-based CR adaptation
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.7 + 0.2 * (ranks/NP);
    
    % 5. Mutation with directional components
    diff1 = popdecs(r(:,1), :) - popdecs(r(:,2), :);
    diff2 = popdecs(r(:,3), :) - popdecs(r(:,4), :);
    mutants = x_best + F.*diff1 + 0.5*diff2;
    
    % 6. Crossover with jrand enforcement
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % 7. Create offspring with boundary handling
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Final clamping
    offspring = max(min(offspring, ub_rep), lb_rep);
end