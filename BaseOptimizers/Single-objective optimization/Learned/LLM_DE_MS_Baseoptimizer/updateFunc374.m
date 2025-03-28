% MATLAB Code
function [offspring] = updateFunc374(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Calculate feasibility ratio
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask) / NP;
    
    % Normalize constraints and fitness
    min_cons = min(cons);
    max_cons = max(cons);
    norm_cons = (cons - min_cons) / (max_cons - min_cons + eps);
    
    min_fit = min(popfits);
    max_fit = max(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    % Identify best solutions
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    % Identify best feasible solution
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
    
    % Ensure r1 != i and r2 != i != r1
    mask = r1 == (1:NP)';
    r1(mask) = mod(r1(mask) + randi(NP-1, sum(mask), 1), NP) + 1;
    mask = r2 == (1:NP)' | r2 == r1;
    r2(mask) = mod(r2(mask) + randi(NP-2, sum(mask), 1), NP) + 1;
    
    % Compute direction vectors
    diff_feas = best_feas - popdecs;
    diff_best = best - popdecs;
    diff_div = popdecs(r1, :) - popdecs(r2, :);
    
    % Compute adaptive weights
    F1 = 0.5 * (1 - rho) .* (1 - norm_cons);
    F2 = 0.3 * rho .* (1 - norm_fits);
    F3 = 0.2 * ones(NP, 1);
    
    % Apply weights
    F1 = repmat(F1, 1, D);
    F2 = repmat(F2, 1, D);
    F3 = repmat(F3, 1, D);
    
    % Generate offspring
    offspring = popdecs + F1 .* diff_feas + F2 .* diff_best + F3 .* diff_div;
    
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