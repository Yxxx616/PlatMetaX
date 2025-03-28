% MATLAB Code
function [offspring] = updateFunc1211(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Rank-based scaling factor
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_idx) = (1:NP)';
    F = 0.4 + 0.6*(ranks/NP);
    
    % 2. Constraint-aware direction weights
    cv_abs = abs(cons);
    max_cv = max(cv_abs);
    w = 1./(1 + exp(-5*cv_abs/(max_cv + eps)));
    
    % 3. Select reference points
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Feasible centroid
    feasible = cons <= 0;
    if any(feasible)
        x_feas = mean(popdecs(feasible, :), 1);
    else
        x_feas = mean(popdecs, 1);
    end
    
    % 4. Generate direction vectors
    term1 = bsxfun(@minus, x_best, popdecs);
    term2 = bsxfun(@minus, x_feas, popdecs);
    direction = bsxfun(@times, (1-w), term1) + bsxfun(@times, w, term2);
    
    % 5. Generate enhanced difference vectors
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
    diff_avg = (diff1 + diff2 + diff3)/3;
    
    % 6. Mutation
    mutants = popdecs + bsxfun(@times, F, direction) + 0.7*diff_avg;
    
    % 7. Adaptive crossover
    CR = 0.1 + 0.7*(1 - ranks/NP);
    mask = rand(NP, D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with bounce-back
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = lb(lb_mask) + rand(sum(lb_mask(:)),1).*(popdecs(lb_mask) - lb(lb_mask));
    offspring(ub_mask) = ub(ub_mask) - rand(sum(ub_mask(:)),1).*(ub(ub_mask) - popdecs(ub_mask));
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end