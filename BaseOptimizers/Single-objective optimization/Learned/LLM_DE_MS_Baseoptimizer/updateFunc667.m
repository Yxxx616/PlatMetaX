% MATLAB Code
function [offspring] = updateFunc667(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Constrained ranking
    penalty = 1e6 * max(0, cons);
    combined = popfits + penalty;
    [~, rank_order] = sort(combined);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    norm_ranks = ranks / NP;
    
    % 3. Normalized constraint violations
    norm_cons = abs(cons) / (max(abs(cons)) + eps);
    
    % 4. Compute weights for direction vectors
    tau = 0.1 * NP;
    weights = exp(-ranks/tau);
    weights = weights / sum(weights);
    
    % 5. Generate random indices (ensuring distinct)
    rand_idx = zeros(NP,4);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available),4));
    end
    r1 = rand_idx(:,1); r2 = rand_idx(:,2);
    r3 = rand_idx(:,3); r4 = rand_idx(:,4);
    
    % 6. Compute direction vectors
    elite_dir = repmat(elite, NP, 1) - popdecs;
    rand_dir = popdecs(r1,:) - popdecs(r2,:) + popdecs(r3,:) - popdecs(r4,:);
    
    % Weighted cons direction
    cons_dir = zeros(NP, D);
    for i = 1:NP
        weighted_sum = zeros(1,D);
        for k = 1:NP
            weighted_sum = weighted_sum + weights(k)*(popdecs(k,:) - popdecs(i,:));
        end
        cons_dir(i,:) = weighted_sum * (1 + norm_cons(i));
    end
    
    % 7. Adaptive scaling factors
    F_elite = 0.5 * (1 - norm_ranks);
    F_rand = 0.3 + 0.2 * randn(NP,1);
    F_cons = 0.2 * (1 - norm_cons);
    
    % 8. Mutation operation
    mutant = popdecs + F_elite.*elite_dir + F_rand.*rand_dir + F_cons.*cons_dir;
    
    % 9. Adaptive crossover
    CR = 0.1 + 0.8 * (1 - norm_ranks);
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | (repmat(1:D, NP, 1) == repmat(j_rand, 1, D));
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 10. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % 11. Final clamping
    offspring = min(max(offspring, lb_rep), ub_rep);
end