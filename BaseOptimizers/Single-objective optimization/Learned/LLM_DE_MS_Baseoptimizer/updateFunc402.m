% MATLAB Code
function [offspring] = updateFunc402(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask) / NP;
    
    % Normalize constraints and fitness
    min_cons = min(cons); max_cons = max(cons);
    norm_cons = (cons - min_cons) / (max_cons - min_cons + 1e-12);
    
    min_fits = min(popfits); max_fits = max(popfits);
    norm_fits = (popfits - min_fits) / (max_fits - min_fits + 1e-12);
    
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
    
    % Generate random indices without repetition
    r1 = zeros(NP,1); r2 = zeros(NP,1);
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        r1(i) = candidates(randi(length(candidates)));
        remaining = setdiff(candidates, r1(i));
        r2(i) = remaining(randi(length(remaining)));
    end
    
    % Compute direction vectors
    diff_feas = repmat(best_feas, NP, 1) - popdecs;
    diff_best = repmat(best, NP, 1) - popdecs;
    diff_div = popdecs(r1, :) - popdecs(r2, :);
    
    % Adaptive weights (enhanced version)
    w_f = 0.9 * (1 - rho) * (1 - norm_cons);
    w_b = 0.7 * rho * (1 - norm_fits);
    w_d = 0.5 * rho * norm_fits;
    
    % Expand weights to D dimensions
    w_f = repmat(w_f, 1, D);
    w_b = repmat(w_b, 1, D);
    w_d = repmat(w_d, 1, D);
    
    % Generate offspring with adaptive perturbation
    offspring = popdecs + w_f.*diff_feas + w_b.*diff_best + w_d.*diff_div;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
               (2*lb_rep - offspring) .* mask_low + ...
               (2*ub_rep - offspring) .* mask_high;
    
    % Adaptive perturbation based on feasibility ratio
    perturb_scale = 0.05 * (1 - rho);
    rand_perturb = perturb_scale * (rand(NP,D) - 0.5) .* (ub_rep - lb_rep);
    offspring = offspring + rand_perturb;
    
    % Final boundary check
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Ensure no NaN values
    offspring(isnan(offspring)) = lb_rep(isnan(offspring)) + ...
                                rand(sum(sum(isnan(offspring))),1) .* ...
                                (ub_rep(isnan(offspring)) - lb_rep(isnan(offspring)));
end