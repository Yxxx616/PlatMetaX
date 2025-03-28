% MATLAB Code
function [offspring] = updateFunc371(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Calculate feasibility ratio and identify feasible solutions
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask) / NP;
    
    % Normalize constraint violations (0 to 1)
    min_cons = min(cons);
    max_cons = max(cons);
    norm_cons = (cons - min_cons) / (max_cons - min_cons + eps);
    
    % Normalize fitness values (0 to 1)
    min_fit = min(popfits);
    max_fit = max(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    % Select best overall and best feasible solutions
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    if any(feasible_mask)
        [~, best_feas_idx] = min(popfits(feasible_mask));
        best_feas = popdecs(feasible_mask, :);
        best_feas = best_feas(best_feas_idx, :);
    else
        best_feas = best;
    end
    
    % Generate random indices for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    
    % Compute direction vectors
    diff_feas = repmat(best_feas, NP, 1) - popdecs;
    diff_best = repmat(best, NP, 1) - popdecs;
    diff_rand = popdecs(r1, :) - popdecs(r2, :);
    
    % Compute adaptive weights
    w_feas = 0.5 * (1 - rho) .* (1 - norm_cons);
    w_fit = 0.5 * rho .* (1 - norm_fits);
    w_rand = 0.2 * (1 - w_feas - w_fit);
    
    % Apply weights to direction vectors
    w_feas = repmat(w_feas, 1, D);
    w_fit = repmat(w_fit, 1, D);
    w_rand = repmat(w_rand, 1, D);
    
    % Generate offspring with adaptive mutation
    offspring = popdecs + w_feas .* diff_feas + w_fit .* diff_best + w_rand .* diff_rand;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring = offspring .* ~(below | above) + ...
                (2*lb_rep - offspring) .* below + ...
                (2*ub_rep - offspring) .* above;
    
    % Final boundary check
    offspring = min(max(offspring, lb_rep), ub_rep);
end