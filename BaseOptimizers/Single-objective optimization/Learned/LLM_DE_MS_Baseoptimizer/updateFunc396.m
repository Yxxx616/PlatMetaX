% MATLAB Code
function [offspring] = updateFunc396(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask) / NP;
    
    % Normalize constraints and fitness
    norm_cons = (cons - min(cons)) ./ (max(cons) - min(cons) + eps);
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    
    % Identify best solutions
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    % Best feasible solution
    if any(feasible_mask)
        [~, best_feas_idx] = min(popfits(feasible_mask));
        best_feas = popdecs(feasible_mask, :);
        best_feas = best_feas(best_feas_idx, :);
    else
        [~, least_infeas_idx] = min(cons);
        best_feas = popdecs(least_infeas_idx, :);
    end
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    for i = 1:NP
        while r1(i) == i, r1(i) = randi(NP); end
        while r2(i) == i || r2(i) == r1(i), r2(i) = randi(NP); end
    end
    
    % Compute direction vectors
    diff_feas = repmat(best_feas, NP, 1) - popdecs;
    diff_best = repmat(best, NP, 1) - popdecs;
    diff_div = popdecs(r1, :) - popdecs(r2, :);
    
    % Adaptive weights
    F1 = 0.7 * (1 - rho) * (1 - norm_cons);
    F2 = 0.5 * rho * (1 - norm_fits);
    F3 = 0.3 * rho * norm_fits;
    
    % Expand weights
    F1 = repmat(F1, 1, D);
    F2 = repmat(F2, 1, D);
    F3 = repmat(F3, 1, D);
    
    % Generate offspring
    offspring = popdecs + F1.*diff_feas + F2.*diff_best + F3.*diff_div;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
               (2*lb_rep - offspring) .* mask_low + ...
               (2*ub_rep - offspring) .* mask_high;
    
    % Final clamping
    offspring = max(min(offspring, ub_rep), lb_rep);
end