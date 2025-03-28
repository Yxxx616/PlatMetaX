% MATLAB Code
function [offspring] = updateFunc912(popdecs, popfits, cons)
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
    r = zeros(NP,7);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 7));
    end
    
    % 3. Compute adaptive parameters
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    c_max = max(abs(cons)) + eps;
    
    norm_fits = (popfits - f_min) / f_range;
    norm_cons = abs(cons) / c_max;
    
    % Adaptive weights
    w_exploit = 0.6 * (1 - norm_fits);
    w_explore = 0.2 * norm_fits;
    w_const = 0.2 * norm_cons;
    w_diverse = 0.2 * (1 - norm_cons);
    
    % Normalize weights
    total = w_exploit + w_explore + w_const + w_diverse;
    w_exploit = w_exploit ./ total;
    w_explore = w_explore ./ total;
    w_const = w_const ./ total;
    w_diverse = w_diverse ./ total;
    
    % 4. Mutation components
    % Elite-guided
    v1 = x_best + 0.8 * (popdecs(r(:,1),:) - popdecs(r(:,2),:));
    
    % Constraint-aware
    omega = 1 - norm_cons(r(:,3));
    v2 = popdecs(r(:,3),:) + 1.2 * omega.*(popdecs(r(:,4),:) - popdecs(r(:,5),:));
    
    % Diversity
    centroid = mean(popdecs, 1);
    sigma = 0.1 * (ub - lb);
    v3 = centroid + sigma .* randn(NP,D);
    
    % Local search
    v4 = popdecs + 0.5*(x_best - popdecs) + 0.5*(popdecs(r(:,6),:) - popdecs(r(:,7),:));
    
    % 5. Composite mutation
    mutants = w_exploit.*v1 + w_const.*v2 + w_diverse.*v3 + w_explore.*v4;
    
    % 6. Adaptive CR
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    CR = 0.3 + 0.6 * (ranks/NP);
    
    % 7. Crossover with jitter
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 8. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Final clamping
    offspring = max(min(offspring, ub_rep), lb_rep);
end