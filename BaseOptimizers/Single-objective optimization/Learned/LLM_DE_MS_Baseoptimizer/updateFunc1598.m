% MATLAB Code
function [offspring] = updateFunc1598(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + 1e6*max(0, cons));
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Compute adaptive parameters
    f_min = min(popfits);
    f_max = max(popfits);
    cons_min = min(cons);
    cons_max = max(cons);
    
    F = 0.5 * (1 + (popfits - f_min) ./ (f_max - f_min + eps));
    lambda = 1 + (cons - cons_min) ./ (cons_max - cons_min + eps);
    CR = 0.9 - 0.5 * (popfits - f_min) ./ (f_max - f_min + eps);
    
    % 3. Generate random indices (vectorized)
    [~, idx] = sort(rand(NP, NP), 2);
    r1 = idx(:,1); r2 = idx(:,2); r3 = idx(:,3); r4 = idx(:,4);
    
    % 4. Create mutation vectors with adaptive scaling
    mutation = elite + F.*(popdecs(r1,:) - popdecs(r2,:)) + ...
               lambda.*(popdecs(r3,:) - popdecs(r4,:));
    
    % 5. Crossover with adaptive CR
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    offspring = popdecs .* (~mask) + mutation .* mask;
    
    % 6. Boundary handling with midpoint repair
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = (popdecs(below_lb) + lb_rep(below_lb)) / 2;
    offspring(above_ub) = (popdecs(above_ub) + ub_rep(above_ub)) / 2;
    
    % 7. Controlled random perturbation
    perturb_mask = rand(NP,D) < 0.05;
    perturb_amount = rand(NP,D) .* (ub_rep - lb_rep) * 0.1;
    offspring = offspring + perturb_mask .* perturb_amount;
    offspring = min(max(offspring, lb_rep), ub_rep);
end