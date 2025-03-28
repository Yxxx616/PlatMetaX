% MATLAB Code
function [offspring] = updateFunc160(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize constraint violations and calculate ranks
    abs_cons = abs(cons);
    norm_cons = abs_cons / (max(abs_cons) + eps);
    [~, fit_rank_idx] = sort(popfits);
    [~, cons_rank_idx] = sort(abs_cons);
    
    % 2. Calculate weights and adaptive parameters
    fit_ranks = zeros(NP,1);
    fit_ranks(fit_rank_idx) = (1:NP)';
    cons_ranks = zeros(NP,1);
    cons_ranks(cons_rank_idx) = (1:NP)';
    
    w = (NP - fit_ranks) / NP;  % Fitness-based weights
    F = 0.5 + 0.3 * tanh(1 - norm_cons);  % Constraint-aware scaling
    CR = 0.7 + 0.2 * cos(pi * cons_ranks / (2 * NP));  % Adaptive CR
    
    % 3. Identify best individual (considering constraints)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = popdecs(feasible_mask(best_idx),:);
    else
        [~, best_idx] = min(abs_cons);
        x_best = popdecs(best_idx,:);
    end
    
    % 4. Generate random indices (vectorized)
    rand_idx = zeros(NP, 2);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available), 2));
    end
    r1 = rand_idx(:,1); 
    r2 = rand_idx(:,2);
    
    % 5. Hybrid mutation with adaptive weights
    term1 = repmat(F .* w, 1, D) .* (repmat(x_best, NP, 1) - popdecs);
    term2 = repmat(F .* (1-w), 1, D) .* (popdecs(r1,:) - popdecs(r2,:));
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
    
    % Reflection for boundary violations
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Final projection to ensure within bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Additional check for extreme cases
    extreme = (offspring < lb_rep-100) | (offspring > ub_rep+100);
    offspring(extreme) = lb_rep(extreme) + rand(sum(extreme(:)),1).*...
                         (ub_rep(extreme) - lb_rep(extreme));
end