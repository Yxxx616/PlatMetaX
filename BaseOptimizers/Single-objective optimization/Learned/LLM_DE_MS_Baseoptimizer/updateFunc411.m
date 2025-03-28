% MATLAB Code
function [offspring] = updateFunc411(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask) / NP;
    
    % Normalize constraints and fitness with protection
    min_cons = min(cons);
    max_cons = max(cons);
    norm_cons = (cons - min_cons) / (max_cons - min_cons + eps);
    
    min_fit = min(popfits);
    max_fit = max(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    % Identify best solution (feasible first)
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        best = popdecs(feasible_mask(best_idx), :);
    else
        [~, best_idx] = min(cons);
        best = popdecs(best_idx, :);
    end
    
    % Calculate feasible center
    if any(feasible_mask)
        feas_center = mean(popdecs(feasible_mask, :), 1);
    else
        feas_center = best;
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
    
    % Direction vectors
    feas_dir = repmat(feas_center, NP, 1) - popdecs;
    elite_dir = repmat(best, NP, 1) - popdecs;
    div_dir = popdecs(r1, :) - popdecs(r2, :);
    
    % Adaptive weights
    w_f = 0.7 * (1 - rho) .* (1 - norm_cons);
    w_e = 0.5 * rho .* (1 - norm_fits);
    w_d = 0.3 * rho .* norm_fits;
    
    % Expand weights to D dimensions
    w_f = repmat(w_f, 1, D);
    w_e = repmat(w_e, 1, D);
    w_d = repmat(w_d, 1, D);
    
    % Adaptive mutation with dynamic scaling
    sigma = 0.1 * (1 - rho) * (ub - lb);
    perturb = randn(NP, D) .* repmat(sigma, NP, 1);
    offspring = popdecs + w_f.*feas_dir + w_e.*elite_dir + w_d.*div_dir + perturb;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
               (lb_rep + abs(offspring - lb_rep)) .* mask_low + ...
               (ub_rep - abs(offspring - ub_rep)) .* mask_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end