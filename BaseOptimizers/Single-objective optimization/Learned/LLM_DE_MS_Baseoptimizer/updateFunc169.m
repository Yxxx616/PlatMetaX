% MATLAB Code
function [offspring] = updateFunc169(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Constraint-aware weighting
    penalty = max(0, cons);
    w = 1 ./ (1 + penalty.^2);
    w = w / max(w); % Normalize to [0,1]
    
    % 2. Identify best individual (feasible preferred)
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible(best_idx),:);
    else
        [~, best_idx] = min(penalty);
        x_best = popdecs(best_idx,:);
    end
    
    % 3. Generate random indices (vectorized)
    rand_idx = zeros(NP, 4);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available), 4));
    end
    r1 = rand_idx(:,1); r2 = rand_idx(:,2); 
    r3 = rand_idx(:,3); r4 = rand_idx(:,4);
    
    % 4. Adaptive mutation
    F = 0.5 + 0.5 * w;
    v = repmat(x_best, NP, 1) + ...
        F .* (popdecs(r1,:) - popdecs(r2,:)) + ...
        (1-w) .* (popdecs(r3,:) - popdecs(r4,:));
    
    % 5. Dynamic crossover
    CR = 0.9 - 0.4 * (w - min(w)) / (max(w) - min(w) + eps);
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % 6. Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    alpha = 0.2 + 0.6 * rand(NP, D);
    offspring(below_lb) = lb_rep(below_lb) + alpha(below_lb) .* ...
                        (offspring(below_lb) - lb_rep(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - alpha(above_ub) .* ...
                        (offspring(above_ub) - ub_rep(above_ub));
    
    % Final projection to bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end