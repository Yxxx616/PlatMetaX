% MATLAB Code
function [offspring] = updateFunc665(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with feasibility consideration
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Rank solutions based on combined fitness and constraints
    combined = popfits + 1e6*max(0, cons);
    [~, rank_order] = sort(combined);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    
    % 3. Normalized constraint violations
    norm_cons = abs(cons) / (max(abs(cons)) + eps);
    
    % 4. Adaptive scaling factors
    F_base = 0.5 * (1 - ranks/NP);
    F_cons = 0.3 * (1 - norm_cons);
    F_rand = 0.2 * randn(NP,1);
    F = F_base + F_cons + F_rand;
    
    % 5. Generate random indices (ensuring distinct)
    rand_idx = zeros(NP,4);
    for i=1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available),4));
    end
    
    % 6. Mutation operation
    a = rand_idx(:,1); b = rand_idx(:,2);
    c = rand_idx(:,3); d = rand_idx(:,4);
    
    mutant = popdecs + ...
        F.*(repmat(elite, NP, 1) - popdecs) + ...
        F.*(popdecs(a,:) - popdecs(b,:)) + ...
        0.1*(popdecs(c,:) - popdecs(d,:));
    
    % 7. Adaptive crossover
    CR = 0.1 + 0.8*(1 - ranks/NP);
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | (repmat(1:D, NP, 1) == repmat(j_rand, 1, D));
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 8. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % 9. Final check with clamping if reflection fails
    offspring = min(max(offspring, lb_rep), ub_rep);
end