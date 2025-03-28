% MATLAB Code
function [offspring] = updateFunc377(popdecs, popfits, cons)
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
    
    % Identify best solution overall
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
    
    % Generate random indices ensuring diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    
    for i = 1:NP
        while r1(i) == i
            r1(i) = randi(NP);
        end
        while r2(i) == i || r2(i) == r1(i)
            r2(i) = randi(NP);
        end
    end
    
    % Compute direction vectors
    diff_feas = repmat(best_feas, NP, 1) - popdecs;
    diff_best = repmat(best, NP, 1) - popdecs;
    diff_div = popdecs(r1, :) - popdecs(r2, :);
    
    % Compute adaptive weights
    F1 = 0.8 * (1 - rho) .* (1 - norm_cons);
    F2 = 0.5 * rho .* (1 - norm_fits);
    F3 = 0.3 * rand(NP, 1);
    
    % Apply weights with vectorization
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