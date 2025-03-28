% MATLAB Code
function [offspring] = updateFunc1600(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution considering both fitness and constraints
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx), :);
    else
        [~, elite_idx] = min(popfits + 1e6*max(0, cons));
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Compute fitness-based ranks and scaling factors
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    F = 0.4 + 0.4 * (ranks / NP);
    
    % 3. Constraint-aware scaling
    cons_min = min(cons);
    cons_max = max(cons);
    lambda = 1 + (cons - cons_min) / (cons_max - cons_min + eps);
    
    % 4. Generate random indices (vectorized)
    [~, idx] = sort(rand(NP, NP), 2);
    r1 = idx(:,1); r2 = idx(:,2); r3 = idx(:,3); r4 = idx(:,4);
    
    % 5. Create mutation vectors with adaptive scaling
    mutation = elite(ones(NP,1), :) + ...
               F.*(popdecs(r1,:) - popdecs(r2,:)) + ...
               lambda.*F.*(popdecs(r3,:) - popdecs(r4,:));
    
    % 6. Adaptive CR based on fitness ranks
    CR = 0.9 - 0.4 * (ranks / NP);
    
    % 7. Crossover with opposition-based learning
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Opposition-based components
    opposition = lb + ub - popdecs;
    offspring = popdecs .* (~mask) + mutation .* mask;
    opposition_mask = rand(NP,D) > 0.5;
    offspring = offspring .* (~opposition_mask) + opposition .* opposition_mask;
    
    % 8. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    offspring = min(max(offspring, lb_rep), ub_rep);
    
    % 9. Small random perturbation for diversity
    perturb_mask = rand(NP,D) < 0.05;
    perturb_amount = randn(NP,D) .* (ub_rep - lb_rep) * 0.01;
    offspring = offspring + perturb_mask .* perturb_amount;
    offspring = min(max(offspring, lb_rep), ub_rep);
end