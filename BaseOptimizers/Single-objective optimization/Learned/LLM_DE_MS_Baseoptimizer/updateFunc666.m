% MATLAB Code
function [offspring] = updateFunc666(popdecs, popfits, cons)
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
    
    % 2. Rank solutions based on constrained fitness
    penalty = 1e6 * max(0, cons);
    combined = popfits + penalty;
    [~, rank_order] = sort(combined);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    norm_ranks = ranks / NP;
    
    % 3. Normalized constraint violations
    norm_cons = abs(cons) / (max(abs(cons)) + eps);
    
    % 4. Generate random indices (ensuring distinct)
    rand_idx = zeros(NP,4);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available),4));
    end
    a = rand_idx(:,1); b = rand_idx(:,2);
    c = rand_idx(:,3); d = rand_idx(:,4);
    
    % 5. Adaptive scaling factors
    F1 = 0.4 * (1 - norm_ranks);
    F2 = 0.3 + 0.2 * randn(NP,1);
    F3 = 0.1 * (1 - norm_cons);
    xi = norm_cons;
    
    % 6. Mutation operation
    elite_term = F1 .* (repmat(elite, NP, 1) - popdecs);
    diff_term = F2 .* (popdecs(a,:) - popdecs(b,:));
    cons_term = F3 .* (popdecs(c,:) - popdecs(d,:)) .* (1 + repmat(xi,1,D));
    
    mutant = popdecs + elite_term + diff_term + cons_term;
    
    % 7. Adaptive crossover
    CR = 0.1 + 0.8 * (1 - norm_ranks);
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
    
    % 9. Final clamping
    offspring = min(max(offspring, lb_rep), ub_rep);
end