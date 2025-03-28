% MATLAB Code
function [offspring] = updateFunc1209(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Rank-based fitness scaling
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_idx) = (1:NP)';
    F1 = 0.5 + 0.3 * (ranks/NP);
    F3 = 0.8 - 0.4 * (ranks/NP);
    
    % 2. Constraint-based scaling
    cv_abs = abs(cons);
    max_cv = max(cv_abs);
    F2 = 0.3 * (1 - cv_abs/(max_cv + eps));
    
    % 3. Select reference points
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Feasible and infeasible centroids
    feasible = cons <= 0;
    if any(feasible)
        x_feas = mean(popdecs(feasible, :), 1);
    else
        x_feas = mean(popdecs, 1);
    end
    
    infeas = ~feasible;
    if any(infeas)
        x_inf = mean(popdecs(infeas, :), 1);
    else
        x_inf = zeros(1, D);
    end
    
    % 4. Generate multiple difference vectors
    idx = (1:NP)';
    r1 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r2 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r3 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r4 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r5 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r6 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r7 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r8 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    diff3 = popdecs(r5,:) - popdecs(r6,:);
    diff4 = popdecs(r7,:) - popdecs(r8,:);
    diff_sum = diff1 + diff2 + diff3 + diff4;
    
    % 5. Mutation
    term1 = bsxfun(@times, F1, x_best - popdecs);
    term2 = bsxfun(@times, F2, x_feas - x_inf);
    term3 = bsxfun(@times, F3/4, diff_sum);
    
    mutants = popdecs + term1 + term2 + term3;
    
    % 6. Adaptive crossover
    CR = 0.1 + 0.7 * (1 - ranks/NP);
    mask = rand(NP, D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end