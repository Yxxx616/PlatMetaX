% MATLAB Code
function [offspring] = updateFunc662(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Rank calculation (considering both fitness and constraints)
    combined = popfits + 1e6*max(0, cons); % Penalty for infeasible
    [~, rank_order] = sort(combined);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    
    % 3. Adaptive factors
    F_base = 0.4 * (1 + cos(pi * ranks / NP));
    max_cv = max(abs(cons));
    F_cv = 0.3 * tanh(1 - abs(cons)./(max_cv + eps));
    F_rank = 0.2 * (1 - ranks/NP) .* randn(NP,1);
    
    % 4. Vectorized random indices
    [~, rand_idx] = sort(rand(NP, NP), 2);
    mask = rand_idx ~= repmat((1:NP)', 1, NP);
    rand_idx = rand_idx .* mask;
    rand_idx(rand_idx == 0) = 1;
    rand_idx = rand_idx(:,1:4);
    
    % 5. Mutation
    a = rand_idx(:,1); b = rand_idx(:,2);
    c = rand_idx(:,3); d = rand_idx(:,4);
    
    mutant = popdecs + ...
        F_base.*(repmat(elite, NP, 1) - popdecs) + ...
        F_cv.*(popdecs(a,:) - popdecs(b,:)) + ...
        F_rank.*(popdecs(c,:) - popdecs(d,:));
    
    % 6. Adaptive crossover
    CR = 0.1 + 0.8*(1 - ranks/NP).^2;
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | (repmat(1:D, NP, 1) == repmat(j_rand, 1, D));
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 7. Enhanced boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection with damping factor
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = lb_rep(below_lb) + 0.5*rand(sum(below_lb(:)),1).*...
        (ub_rep(below_lb)-lb_rep(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - 0.5*rand(sum(above_ub(:)),1).*...
        (ub_rep(above_ub)-lb_rep(above_ub));
    
    % Final check with random reinitialization
    still_invalid = (offspring < lb_rep) | (offspring > ub_rep);
    offspring(still_invalid) = lb_rep(still_invalid) + ...
        rand(sum(still_invalid(:)),1).*(ub_rep(still_invalid)-lb_rep(still_invalid));
end