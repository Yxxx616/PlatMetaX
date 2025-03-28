% MATLAB Code
function [offspring] = updateFunc654(popdecs, popfits, cons)
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
    F1 = 0.4 * (1 - ranks/NP).^1.2;
    F2 = 0.3 * exp(-cons./(sigma_cv + eps));
    F3 = 0.5 * (1 - ranks/NP).^0.8;
    F4 = F3;
    
    % 4. Generate random indices matrix
    idx_mat = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        perm = randperm(length(candidates), 4);
        idx_mat(i,:) = candidates(perm);
    end
    
    % 5. Vectorized mutation
    mutant = zeros(NP, D);
    for i = 1:NP
        r1 = idx_mat(i,1); r2 = idx_mat(i,2); 
        r3 = idx_mat(i,3); r4 = idx_mat(i,4);
        
        mutant(i,:) = popdecs(i,:) + ...
            F1(i)*(elite - popdecs(i,:)) + ...
            F2(i)*(feas_best - popdecs(i,:)) + ...
            F3(i)*(popdecs(r1,:) - popdecs(r2,:)) + ...
            F4(i)*(popdecs(r3,:) - popdecs(r4,:));
    end
    
    % 6. Crossover with adaptive CR
    CR = 0.1 + 0.8*(1 - ranks/NP).^1.5;
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