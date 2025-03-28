% MATLAB Code
function [offspring] = updateFunc389(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Calculate feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask) / NP;
    
    % Normalize constraints and fitness (0 to 1)
    norm_cons = (cons - min(cons)) ./ (max(cons) - min(cons) + eps);
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps);
    
    % Identify best solutions
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    % Find best feasible solution
    if any(feasible_mask)
        [~, best_feas_idx] = min(popfits(feasible_mask));
        best_feas = popdecs(feasible_mask, :);
        best_feas = best_feas(best_feas_idx, :);
    else
        [~, least_infeas_idx] = min(cons);
        best_feas = popdecs(least_infeas_idx, :);
    end
    
    % Generate random indices ensuring r1 ≠ r2 ≠ i
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
    
    % Compute adaptive weights
    Ff = 0.7 * (1 - rho) .* (1 - norm_cons);
    Fb = 0.5 * rho .* (1 - norm_fits);
    Fd = 0.3 * rand(NP, 1);
    
    % Expand weights for vectorized operations
    Ff = repmat(Ff, 1, D);
    Fb = repmat(Fb, 1, D);
    Fd = repmat(Fd, 1, D);
    
    % Generate offspring with adaptive mutation
    offspring = popdecs + Ff .* diff_feas + Fb .* diff_best + Fd .* diff_div;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
               (2*lb_rep - offspring) .* mask_low + ...
               (2*ub_rep - offspring) .* mask_high;
    
    % Final clamping to ensure strict bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Small random perturbation to maintain diversity
    perturb = 0.05 * (ub - lb) .* randn(NP, D);
    offspring = offspring + perturb;
    offspring = max(min(offspring, ub_rep), lb_rep);
end