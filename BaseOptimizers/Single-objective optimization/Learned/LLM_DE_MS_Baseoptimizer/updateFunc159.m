% MATLAB Code
function [offspring] = updateFunc159(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize constraint violations and calculate fitness ranks
    norm_cons = abs(cons) / (max(abs(cons)) + eps);
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_idx) = (1:NP)'/NP;
    
    % 2. Adaptive parameters based on constraints and ranks
    F = 0.5 * (1 + tanh(1 - norm_cons)) .* (1 - ranks);
    CR = 0.9 - 0.3 * norm_cons;
    
    % 3. Identify best individual (considering constraints)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = popdecs(feasible_mask(best_idx),:);
    else
        [~, best_idx] = min(cons);
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
    
    % 5. Current-to-best mutation with adaptive F
    v = popdecs + F.*(x_best - popdecs) + F.*(popdecs(r1,:) - popdecs(r2,:));
    
    % 6. Crossover with adaptive CR
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % 7. Enhanced boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Midpoint repair for severe violations
    severe_violation = (offspring < lb_rep-10) | (offspring > ub_rep+10);
    offspring(severe_violation) = (offspring(severe_violation) + ...
        (lb_rep(severe_violation) + ub_rep(severe_violation))/2);
    
    % Random repair for moderate violations
    moderate_violation = (offspring < lb_rep & ~severe_violation) | ...
                        (offspring > ub_rep & ~severe_violation);
    rand_vals = rand(sum(moderate_violation(:)),1);
    offspring(moderate_violation) = lb_rep(moderate_violation) + ...
        rand_vals.*(ub_rep(moderate_violation) - lb_rep(moderate_violation));
    
    % Final projection
    offspring = max(min(offspring, ub_rep), lb_rep);
end