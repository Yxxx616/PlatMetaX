% MATLAB Code
function [offspring] = updateFunc1601(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(popfits + 1e6*max(0, cons));
        elite = popdecs(elite_idx,:);
    end
    
    % 2. Compute ranks for fitness and constraints
    [~, fit_rank_order] = sort(popfits);
    fit_ranks = zeros(NP,1);
    fit_ranks(fit_rank_order) = 1:NP;
    
    [~, cons_rank_order] = sort(cons);
    cons_ranks = zeros(NP,1);
    cons_ranks(cons_rank_order) = 1:NP;
    
    % 3. Adaptive scaling factors
    F = 0.5 * (1 + fit_ranks/NP) .* (1 - cons_ranks/NP);
    
    % 4. Constraint-aware mutation
    idx = randperm(NP, 4*NP);
    r1 = reshape(idx(1:NP), [], 1);
    r2 = reshape(idx(NP+1:2*NP), [], 1);
    r3 = reshape(idx(2*NP+1:3*NP), [], 1);
    r4 = reshape(idx(3*NP+1:4*NP), [], 1);
    
    alpha = 0.5;
    mutation = elite(ones(NP,1), :) + ...
               F.*(popdecs(r1,:) - popdecs(r2,:)) + ...
               alpha * tanh(cons).*(popdecs(r3,:) - popdecs(r4,:));
    
    % 5. Dynamic crossover rate
    CR = 0.9 - 0.5 * (fit_ranks + cons_ranks)/(2*NP);
    
    % 6. Crossover with opposition-based learning
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs .* (~mask) + mutation .* mask;
    
    % Opposition-based learning (30% probability)
    opposition = lb + ub - popdecs;
    opposition_mask = rand(NP,D) < 0.3;
    offspring = offspring .* (~opposition_mask) + opposition .* opposition_mask;
    
    % 7. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    offspring = min(max(offspring, lb_rep), ub_rep);
    
    % 8. Small perturbation for diversity (5% probability)
    perturb_mask = rand(NP,D) < 0.05;
    perturb_amount = randn(NP,D) .* (ub_rep - lb_rep) * 0.01;
    offspring = offspring + perturb_mask .* perturb_amount;
    offspring = min(max(offspring, lb_rep), ub_rep);
end