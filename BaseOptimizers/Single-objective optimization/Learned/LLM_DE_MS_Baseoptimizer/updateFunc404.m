% MATLAB Code
function [offspring] = updateFunc404(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask) / NP;
    
    % Normalize constraints and fitness
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + 1e-12);
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + 1e-12);
    
    % Identify best solution
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
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
    while any(r1 == (1:NP)' | r2 == (1:NP)' | r1 == r2)
        bad_idx = find(r1 == (1:NP)' | r2 == (1:NP)' | r1 == r2);
        r1(bad_idx) = randi(NP, numel(bad_idx), 1);
        r2(bad_idx) = randi(NP, numel(bad_idx), 1);
    end
    
    % Compute direction vectors
    feas_dir = repmat(feas_center, NP, 1) - popdecs;
    elite_dir = repmat(best, NP, 1) - popdecs;
    div_dir = popdecs(r1, :) - popdecs(r2, :);
    
    % Adaptive weights
    w_f = 0.8 * (1 - rho) .* (1 - norm_cons);
    w_e = 0.6 * rho .* (1 - norm_fits);
    w_d = 0.4 * rho .* norm_fits;
    
    % Expand weights to D dimensions
    w_f = repmat(w_f, 1, D);
    w_e = repmat(w_e, 1, D);
    w_d = repmat(w_d, 1, D);
    
    % Generate offspring with adaptive perturbation
    perturb = 0.1 * (1 - rho) * (rand(NP,D) - 0.5) .* (ub - lb);
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