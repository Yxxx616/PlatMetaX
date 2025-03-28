% MATLAB Code
function [offspring] = updateFunc381(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Calculate feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask) / NP;
    
    % Normalize constraints and fitness with enhanced robustness
    norm_cons = (cons - min(cons)) ./ (max(cons) - min(cons) + eps);
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    
    % Identify best solutions with improved selection
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    if any(feasible_mask)
        [~, best_feas_idx] = min(popfits(feasible_mask));
        best_feas = popdecs(feasible_mask, :);
        best_feas = best_feas(best_feas_idx, :);
    else
        % If no feasible, use least infeasible
        [~, least_infeas_idx] = min(cons);
        best_feas = popdecs(least_infeas_idx, :);
    end
    
    % Generate random indices with guaranteed distinctness
    r1 = mod((1:NP)' + randi(NP-1, NP, 1), NP) + 1;
    r2 = mod((1:NP)' + randi(NP-2, NP, 1), NP) + 1;
    r2(r2 == r1) = mod(r2(r2 == r1) + 1, NP) + 1;
    
    % Compute direction vectors with vectorization
    diff_feas = best_feas - popdecs;
    diff_best = best - popdecs;
    diff_div = popdecs(r1, :) - popdecs(r2, :);
    
    % Compute adaptive weights with enhanced coefficients
    F1 = 0.8 * (1 - rho) .* (1 - norm_cons);
    F2 = 0.6 * rho .* (1 - norm_fits);
    F3 = 0.5 * rand(NP, 1);
    
    % Expand weights for vectorized operations
    F1 = repmat(F1, 1, D);
    F2 = repmat(F2, 1, D);
    F3 = repmat(F3, 1, D);
    
    % Generate offspring with enhanced mutation
    offspring = popdecs + F1 .* diff_feas + F2 .* diff_best + F3 .* diff_div;
    
    % Improved boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection for out-of-bound solutions
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
               (2*lb_rep - offspring) .* mask_low + ...
               (2*ub_rep - offspring) .* mask_high;
    
    % Final clamping to ensure strict bounds
    offspring = min(max(offspring, lb_rep), ub_rep);
end