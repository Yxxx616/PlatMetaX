% MATLAB Code
function [offspring] = updateFunc447(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify elite solution (best feasible or least infeasible)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Calculate feasible center
    if any(feasible_mask)
        feas_center = mean(popdecs(feasible_mask, :), 1);
    else
        feas_center = mean(popdecs, 1);
    end
    
    % Normalize fitness and constraints
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    
    % Adaptive scaling factors
    F1 = 0.5 * (1 + norm_fits);
    F2 = 0.5 * (1 - norm_cons);
    CR = 0.3 * (1 + norm_cons);
    
    % Expand to D dimensions
    F1 = repmat(F1, 1, D);
    F2 = repmat(F2, 1, D);
    CR = repmat(CR, 1, D);
    
    % Generate random indices (vectorized)
    r1 = zeros(NP, 1);
    r2 = zeros(NP, 1);
    for i = 1:NP
        available = setdiff(1:NP, i);
        r1(i) = available(randi(NP-1));
        available = setdiff(available, r1(i));
        r2(i) = available(randi(NP-2));
    end
    
    % Direction vectors
    elite_dir = repmat(elite, NP, 1) - popdecs;
    feas_dir = repmat(feas_center, NP, 1) - popdecs;
    div_dir = popdecs(r1, :) - popdecs(r2, :);
    
    % Combined mutation
    offspring = popdecs + F1.*elite_dir + F2.*feas_dir + CR.*div_dir;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
        (2*lb_rep - offspring) .* mask_low + ...
        (2*ub_rep - offspring) .* mask_high;
    
    % Final clipping to ensure strict bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end