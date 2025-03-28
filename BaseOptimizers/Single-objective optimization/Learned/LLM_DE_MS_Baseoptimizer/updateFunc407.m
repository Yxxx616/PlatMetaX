% MATLAB Code
function [offspring] = updateFunc407(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask) / NP;
    
    % Normalize constraints and fitness
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + 1e-12);
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + 1e-12);
    
    % Identify best solution (considering feasibility)
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        best = popdecs(feasible_mask(best_idx), :);
    else
        [~, best_idx] = min(cons);
        best = popdecs(best_idx, :);
    end
    
    % Feasible solutions center
    if any(feasible_mask)
        feas_center = mean(popdecs(feasible_mask, :), 1);
    else
        [~, least_infeas_idx] = min(cons);
        feas_center = popdecs(least_infeas_idx, :);
    end
    
    % Generate unique random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    while any(mask)
        r1(mask) = randi(NP, sum(mask), 1);
        r2(mask) = randi(NP, sum(mask), 1);
        mask = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    end
    
    % Compute direction vectors
    feas_dir = repmat(feas_center, NP, 1) - popdecs;
    elite_dir = repmat(best, NP, 1) - popdecs;
    div_dir = popdecs(r1, :) - popdecs(r2, :);
    
    % Adaptive weights (improved version)
    w_f = 0.9 * (1 - rho) .* (1 - norm_cons);
    w_e = 0.7 * rho .* (1 - norm_fits);
    w_d = 0.5 * rho .* norm_fits;
    
    % Expand weights to D dimensions
    w_f = repmat(w_f, 1, D);
    w_e = repmat(w_e, 1, D);
    w_d = repmat(w_d, 1, D);
    
    % Generate offspring with adaptive perturbation
    sigma = 0.2 * (1 - rho) * (ub - lb);
    perturb = randn(NP,D) .* repmat(sigma, NP, 1);
    offspring = popdecs + w_f.*feas_dir + w_e.*elite_dir + w_d.*div_dir + perturb;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
               (2*lb_rep - offspring) .* mask_low + ...
               (2*ub_rep - offspring) .* mask_high;
    
    % Final boundary check
    offspring = max(min(offspring, ub_rep), lb_rep);
end