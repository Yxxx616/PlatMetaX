% MATLAB Code
function [offspring] = updateFunc916(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select best individual
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible(best_idx),:);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx,:);
    end
    
    % 2. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP,4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % 3. Compute adaptive weights
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    c_max = max(abs(cons)) + eps;
    
    w = (popfits - f_min)./f_range + abs(cons)./c_max;
    alpha = 0.5*(1 + cos(pi*w));
    
    % 4. Mutation with directional components
    F1 = 0.8;
    F2 = 0.6;
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    mutants = x_best + F1*diff1 + F2*alpha.*diff2;
    
    % 5. Crossover with adaptive CR
    CR = 0.9 - 0.4*alpha;
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 6. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with midpoint reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = (lb_rep(below) + popdecs(below))/2;
    offspring(above) = (ub_rep(above) + popdecs(above))/2;
    
    % Final clamping
    offspring = max(min(offspring, ub_rep), lb_rep);
end