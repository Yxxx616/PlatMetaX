% MATLAB Code
function [offspring] = updateFunc161(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Constraint-aware ranking and weighting
    abs_cons = abs(cons);
    [~, cons_rank_idx] = sort(abs_cons);
    w = zeros(NP, 1);
    w(cons_rank_idx) = 1 - (1:NP)'/NP;
    
    % 2. Identify best individual considering both fitness and constraints
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_fit_idx] = min(popfits(feasible_mask));
        x_best = popdecs(feasible_mask(best_fit_idx),:);
    else
        [~, best_cons_idx] = min(abs_cons);
        x_best = popdecs(best_cons_idx,:);
    end
    
    % 3. Generate random indices (vectorized)
    rand_idx = zeros(NP, 2);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available), 2));
    end
    r1 = rand_idx(:,1); 
    r2 = rand_idx(:,2);
    
    % 4. Adaptive parameters
    F = 0.4 + 0.4 * exp(-w);
    CR = 0.5 + 0.3 * cos(pi * w);
    
    % 5. Hybrid mutation
    term1 = repmat(F, 1, D) .* (repmat(x_best, NP, 1) - popdecs);
    term2 = repmat(F, 1, D) .* (popdecs(r1,:) - popdecs(r2,:));
    v = popdecs + term1 + term2;
    
    % 6. Crossover with adaptive CR
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % 7. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final projection to ensure within bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Additional randomization for extreme cases
    extreme = (offspring < lb_rep-100) | (offspring > ub_rep+100);
    offspring(extreme) = lb_rep(extreme) + rand(sum(extreme(:)),1).*...
                         (ub_rep(extreme) - lb_rep(extreme));
end