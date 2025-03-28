% MATLAB Code
function [offspring] = updateFunc370(popdecs, popfits, cons)
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
    
    % Combined adaptive weights (higher for better fitness and lower violations)
    w = 0.7 * (1 - norm_fits) + 0.3 * (1 - norm_cons);
    
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
    
    % Compute direction vectors
    diff_feas = best_feas - popdecs;
    diff_best = best - popdecs;
    
    % Adaptive scaling factors
    F_feas = 0.5 * (1 - rho) .* w;
    F_best = 0.5 * rho .* (1 - w);
    sigma = 0.2 * (1 - w);
    
    % Generate offspring with adaptive mutation
    offspring = popdecs + F_feas .* diff_feas + F_best .* diff_best + sigma .* randn(NP, D);
    
    % Boundary handling with random reinitialization
    out_of_bounds = offspring < lb | offspring > ub;
    rand_pos = lb + (ub - lb) .* rand(NP, D);
    offspring = offspring .* ~out_of_bounds + rand_pos .* out_of_bounds;
    
    % Final boundary check
    offspring = min(max(offspring, lb), ub);
end