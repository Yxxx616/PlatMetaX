% MATLAB Code
function [offspring] = updateFunc372(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Calculate feasibility ratio
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask) / NP;
    
    % Normalize constraints [0,1]
    min_cons = min(cons);
    max_cons = max(cons);
    norm_cons = (cons - min_cons) / (max_cons - min_cons + eps);
    
    % Normalize fitness [0,1]
    min_fit = min(popfits);
    max_fit = max(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    % Identify best solutions
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    if any(feasible_mask)
        [~, best_feas_idx] = min(popfits(feasible_mask));
        best_feas = popdecs(feasible_mask, :);
        best_feas = best_feas(best_feas_idx, :);
    else
        best_feas = best;
    end
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = r1 == r2;
    while any(mask)
        r2(mask) = randi(NP, sum(mask), 1);
        mask = r1 == r2;
    end
    
    % Compute direction vectors
    diff_feas = repmat(best_feas, NP, 1) - popdecs;
    diff_best = repmat(best, NP, 1) - popdecs;
    diff_rand = popdecs(r1, :) - popdecs(r2, :);
    
    % Compute adaptive weights
    w_feas = 0.6 * (1 - rho) .* (1 - norm_cons);
    w_best = 0.3 * rho .* (1 - norm_fits);
    w_rand = 0.1 * (1 - w_feas - w_best);
    
    % Apply weights
    w_feas = repmat(w_feas, 1, D);
    w_best = repmat(w_best, 1, D);
    w_rand = repmat(w_rand, 1, D);
    
    % Generate offspring
    offspring = popdecs + w_feas .* diff_feas + w_best .* diff_best + w_rand .* diff_rand;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring = offspring .* ~(below | above) + ...
                (2*lb_rep - offspring) .* below + ...
                (2*ub_rep - offspring) .* above;
    
    % Final boundary check
    offspring = max(min(offspring, ub_rep), lb_rep);
end