% MATLAB Code
function [offspring] = updateFunc914(popdecs, popfits, cons)
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
    
    % 3. Compute adaptive parameters
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    c_max = max(abs(cons)) + eps;
    
    w_f = (popfits - f_min) ./ f_range;
    w_c = abs(cons) ./ c_max;
    alpha = (1 - w_f) ./ (1 + w_c);
    
    % 4. Mutation with adaptive F
    F = 0.5 + 0.3 * alpha;
    x_base = alpha.*x_best + (1-alpha).*popdecs;
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    mutants = x_base + F.*diff1 + F.*diff2;
    
    % 5. Crossover with adaptive CR
    CR = 0.7 - 0.3 * alpha;
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 6. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Final clamping
    offspring = max(min(offspring, ub_rep), lb_rep);
end