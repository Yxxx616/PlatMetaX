% MATLAB Code
function [offspring] = updateFunc162(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Constraint-aware ranking and weighting
    abs_cons = abs(cons);
    [~, cons_rank] = sort(abs_cons);
    w = zeros(NP, 1);
    w(cons_rank) = (1:NP)'/NP;  % Higher weight for better solutions
    
    % 2. Identify best individual (feasible or least infeasible)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = popdecs(feasible_mask(best_idx),:);
    else
        [~, best_idx] = min(abs_cons);
        x_best = popdecs(best_idx,:);
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
    F1 = 0.5 * (1 + exp(-w));
    F2 = 0.3 * (1 - w);
    CR = 0.9 * w + 0.1;
    epsilon = 0.1;  % Fixed for this iteration
    
    % 5. Enhanced mutation with random perturbation
    term1 = repmat(F1, 1, D) .* (repmat(x_best, NP, 1) - popdecs);
    term2 = repmat(F2, 1, D) .* (popdecs(r1,:) - popdecs(r2,:));
    rand_term = epsilon * randn(NP, D);
    v = popdecs + term1 + term2 + rand_term;
    
    % 6. Crossover with adaptive CR
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % 7. Boundary handling with bounce-back
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = lb_rep(below_lb) + rand(sum(below_lb(:)),1) .* ...
                         (popdecs(below_lb) - lb_rep(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - rand(sum(above_ub(:)),1) .* ...
                         (popdecs(above_ub) - ub_rep(above_ub));
    
    % Ensure strict bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end