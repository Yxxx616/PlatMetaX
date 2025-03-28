% MATLAB Code
function [offspring] = updateFunc397(popdecs, popfits, cons)
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
    
    % Adaptive weights (enhanced version)
    F1 = 0.8 * (1 - rho) * (1 - norm_cons);
    F2 = 0.6 * rho * (1 - norm_fits);
    F3 = 0.4 * rho * norm_fits;
    
    % Expand weights
    F1 = repmat(F1, 1, D);
    F2 = repmat(F2, 1, D);
    F3 = repmat(F3, 1, D);
    
    % Generate offspring with enhanced exploration
    offspring = popdecs + F1.*diff_feas + F2.*diff_best + F3.*diff_div;
    
    % Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    % Adaptive reflection factor based on feasibility
    alpha = 1 + rho;
    offspring = offspring .* ~(mask_low | mask_high) + ...
               (alpha*lb_rep - (alpha-1)*offspring) .* mask_low + ...
               (alpha*ub_rep - (alpha-1)*offspring) .* mask_high;
    
    % Final clamping with small random perturbation
    rand_perturb = 0.01 * (rand(NP,D) - 0.5) .* (ub_rep - lb_rep);
    offspring = max(min(offspring + rand_perturb, ub_rep), lb_rep);
end