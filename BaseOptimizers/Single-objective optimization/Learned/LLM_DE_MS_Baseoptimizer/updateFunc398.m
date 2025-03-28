% MATLAB Code
function [offspring] = updateFunc398(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask) / NP;
    
    % Normalize constraints and fitness with enhanced stability
    norm_cons = (cons - min(cons)) ./ (max(cons) - min(cons) + 1e-10);
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + 1e-10);
    
    % Identify best solutions
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    % Best feasible solution with fallback
    if any(feasible_mask)
        [~, best_feas_idx] = min(popfits(feasible_mask));
        best_feas = popdecs(feasible_mask, :);
        best_feas = best_feas(best_feas_idx, :);
    else
        [~, least_infeas_idx] = min(cons);
        best_feas = popdecs(least_infeas_idx, :);
    end
    
    % Generate random indices without loops
    r1 = mod(randperm(NP)' + randi(NP-1, NP, 1), NP) + 1;
    r2 = mod(randperm(NP)' + randi(NP-1, NP, 1), NP) + 1;
    same_idx = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    r1(same_idx) = mod(r1(same_idx) + randi(NP-1, sum(same_idx), 1), NP) + 1;
    r2(same_idx) = mod(r2(same_idx) + randi(NP-1, sum(same_idx), 1), NP) + 1;
    
    % Compute direction vectors
    diff_feas = best_feas - popdecs;
    diff_best = best - popdecs;
    diff_div = popdecs(r1, :) - popdecs(r2, :);
    
    % Enhanced adaptive weights
    F1 = 0.9 * (1 - rho) * (1 - norm_cons);
    F2 = 0.7 * rho * (1 - norm_fits);
    F3 = 0.5 * rho * norm_fits;
    
    % Expand weights
    F1 = repmat(F1, 1, D);
    F2 = repmat(F2, 1, D);
    F3 = repmat(F3, 1, D);
    
    % Generate offspring with adaptive exploration
    offspring = popdecs + F1.*diff_feas + F2.*diff_best + F3.*diff_div;
    
    % Smart boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Adaptive reflection with feasibility consideration
    reflect_factor = 1.2 + 0.8 * rho;
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
               (reflect_factor*lb_rep - (reflect_factor-1)*offspring) .* mask_low + ...
               (reflect_factor*ub_rep - (reflect_factor-1)*offspring) .* mask_high;
    
    % Final clamping with small adaptive perturbation
    perturb_scale = 0.02 * (1 - rho);
    rand_perturb = perturb_scale * (rand(NP,D) - 0.5) .* (ub_rep - lb_rep);
    offspring = max(min(offspring + rand_perturb, ub_rep), lb_rep);
end