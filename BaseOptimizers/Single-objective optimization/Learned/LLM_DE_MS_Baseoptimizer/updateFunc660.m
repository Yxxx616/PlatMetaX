% MATLAB Code
function [offspring] = updateFunc660(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify elite individual
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Calculate ranks and constraint violation parameters
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    max_cv = max(abs(cons));
    
    % 3. Calculate adaptive scaling factors
    F_base = 0.4 + 0.3 * cos(pi * ranks / (2*NP));
    F_cv = 0.3 * tanh(1 - abs(cons)./(max_cv + eps));
    F_rand = 0.3 * rand(NP,1) .* (1 - ranks/NP);
    
    % 4. Generate random indices matrix (vectorized)
    [~, rand_idx] = sort(rand(NP, NP), 2);
    mask = rand_idx ~= repmat((1:NP)', 1, NP);
    rand_idx = rand_idx .* mask;
    rand_idx(rand_idx == 0) = 1;
    rand_idx = rand_idx(:,1:4);
    
    % 5. Vectorized mutation
    a = rand_idx(:,1); b = rand_idx(:,2);
    c = rand_idx(:,3); d = rand_idx(:,4);
    
    mutant = popdecs + ...
        F_base.*(repmat(elite, NP, 1) - popdecs) + ...
        F_cv.*(popdecs(a,:) - popdecs(b,:)) + ...
        F_rand.*(popdecs(c,:) - popdecs(d,:));
    
    % 6. Adaptive crossover
    CR = 0.1 + 0.8*(1 - ranks/NP).^1.5;
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | (repmat(1:D, NP, 1) == repmat(j_rand, 1, D));
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 7. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Random reinitialization if still out of bounds
    still_invalid = (offspring < lb_rep) | (offspring > ub_rep);
    offspring(still_invalid) = lb_rep(still_invalid) + ...
        rand(sum(still_invalid(:)),1).*(ub_rep(still_invalid)-lb_rep(still_invalid));
end