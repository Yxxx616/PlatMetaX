% MATLAB Code
function [offspring] = updateFunc1213(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Rank-based scaling factor with constraint consideration
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_idx) = (1:NP)';
    cv_abs = abs(cons);
    max_cv = max(cv_abs);
    F = 0.1 + 0.5*(1 - ranks/NP).*(1 + cv_abs/(max_cv + eps));
    
    % 2. Hybrid base vector selection
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    feasible = cons <= 0;
    if any(feasible)
        x_feas = mean(popdecs(feasible, :), 1);
    else
        x_feas = mean(popdecs, 1);
    end
    
    alpha = 1./(1 + exp(-10*min(cv_abs)/(max_cv + eps));
    x_base = bsxfun(@times, alpha, x_best) + bsxfun(@times, (1-alpha), x_feas);
    
    % 3. Directional mutation with multiple differences
    idx = (1:NP)';
    r1 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r2 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r3 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r4 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r5 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r6 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    diff3 = popdecs(r5,:) - popdecs(r6,:);
    
    mutants = bsxfun(@plus, x_base, bsxfun(@times, F, diff1) + 0.5*bsxfun(@times, F, diff2 + diff3));
    
    % 4. Constraint-guided crossover
    CR = 0.1 + 0.7*(1 - cv_abs/(max_cv + eps));
    mask = rand(NP, D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 5. Adaptive boundary handling
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = (lb(lb_mask) + popdecs(lb_mask))/2;
    offspring(ub_mask) = (ub(ub_mask) + popdecs(ub_mask))/2;
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end