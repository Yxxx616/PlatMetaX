% MATLAB Code
function [offspring] = updateFunc656(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify elite and feasible best
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
        [~, feas_best_idx] = min(popfits(feasible));
        feas_best = popdecs(temp(feas_best_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
        feas_best = elite;
    end
    
    % 2. Calculate ranks and constraint violation parameters
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = 1:NP;
    sigma_cv = std(cons);
    
    % 3. Calculate adaptive scaling factors
    F1 = 0.6 * (1 - ranks/NP).^1.8;
    F2 = 0.4 * exp(-cons./(sigma_cv + eps));
    F3 = 0.3 * (1 - ranks/NP).^0.5;
    F4 = 0.2 * exp(-cons./(3*sigma_cv + eps));
    
    % 4. Generate random indices matrix (vectorized)
    [~, rand_idx] = sort(rand(NP, NP), 2);
    mask = rand_idx ~= repmat(1:NP, NP, 1);
    rand_idx = rand_idx .* mask;
    rand_idx(rand_idx == 0) = 1;
    rand_idx = rand_idx(:,1:4);
    
    % 5. Vectorized mutation
    r1 = rand_idx(:,1); r2 = rand_idx(:,2);
    r3 = rand_idx(:,3); r4 = rand_idx(:,4);
    
    mutant = popdecs + ...
        F1.*(repmat(elite, NP, 1) - popdecs) + ...
        F2.*(repmat(feas_best, NP, 1) - popdecs) + ...
        F3.*(popdecs(r1,:) - popdecs(r2,:)) + ...
        F4.*(popdecs(r3,:) - popdecs(r4,:));
    
    % 6. Crossover with adaptive CR
    CR = 0.2 + 0.7*(1 - ranks/NP).^1.5;
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | (repmat(1:D, NP, 1) == repmat(j_rand, 1, D));
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 7. Boundary handling with reflection and random reinitialization
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection first
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    
    % Random reinitialization if still out of bounds
    still_invalid = (offspring < lb_rep) | (offspring > ub_rep);
    offspring(still_invalid) = lb_rep(still_invalid) + ...
        rand(sum(still_invalid(:)),1).*(ub_rep(still_invalid)-lb_rep(still_invalid));
end