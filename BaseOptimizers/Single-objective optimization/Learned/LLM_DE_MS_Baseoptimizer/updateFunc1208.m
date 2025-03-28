% MATLAB Code
function [offspring] = updateFunc1208(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select reference points
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
    
    % 2. Compute adaptive scaling factors
    f_min = min(popfits);
    f_max = max(popfits);
    F_f = 0.4 + 0.5 * (popfits - f_min) ./ (f_max - f_min + eps);
    
    c_max = max(abs(cons));
    F_c = 0.5 * (1 + tanh(-abs(cons) ./ (c_max + eps)));
    F = F_f .* F_c;
    
    % 3. Generate mutation vectors with 4 random differences
    idx = (1:NP)';
    r1 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r2 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r3 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r4 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    
    term1 = bsxfun(@times, F, x_best - popdecs);
    term2 = bsxfun(@times, (1 - F), x_feas - x_inf);
    term3 = 0.7 * (popdecs(r1,:) - popdecs(r2,:) + popdecs(r3,:) - popdecs(r4,:));
    
    mutants = popdecs + term1 + term2 + term3;
    
    % 4. Adaptive crossover
    CR = 0.2 + 0.6 * F;
    mask = rand(NP, D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 5. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end