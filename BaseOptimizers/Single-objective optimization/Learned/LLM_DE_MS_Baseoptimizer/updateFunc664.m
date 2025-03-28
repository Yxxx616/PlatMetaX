% MATLAB Code
function [offspring] = updateFunc664(popdecs, popfits, cons)
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
    
    % 2. Normalized fitness and constraint violation
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_cons = abs(cons) / (max(abs(cons)) + eps);
    
    % 3. Rank-based selection pressure
    combined = popfits + 1e6*max(0, cons);
    [~, rank_order] = sort(combined);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    
    % 4. Adaptive scaling factors
    F1 = 0.4 * (1 - tanh(ranks/NP));
    F2 = 0.3 * tanh(1 - norm_cons) .* rand(NP,1);
    F3 = 0.3 * (1 - norm_fits) .* randn(NP,1);
    
    % 5. Vectorized random indices (ensuring distinct)
    rand_idx = zeros(NP,4);
    for i=1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available),4));
    end
    
    % 6. Mutation operation
    a = rand_idx(:,1); b = rand_idx(:,2);
    c = rand_idx(:,3); d = rand_idx(:,4);
    
    mutant = popdecs + ...
        F1.*(repmat(elite, NP, 1) - popdecs) + ...
        F2.*(popdecs(a,:) - popdecs(b,:)) + ...
        F3.*(popdecs(c,:) - popdecs(d,:));
    
    % 7. Adaptive crossover
    CR = 0.2 + 0.7*(1 - ranks/NP).^2;
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
    
    % Reflection with random component
    offspring(below_lb) = lb_rep(below_lb) + 0.4*rand(sum(below_lb(:)),1).*...
        (ub_rep(below_lb)-lb_rep(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - 0.4*rand(sum(above_ub(:)),1).*...
        (ub_rep(above_ub)-lb_rep(above_ub));
    
    % Final check with random reinitialization if needed
    still_invalid = (offspring < lb_rep) | (offspring > ub_rep);
    offspring(still_invalid) = lb_rep(still_invalid) + ...
        rand(sum(still_invalid(:)),1).*(ub_rep(still_invalid)-lb_rep(still_invalid));
end