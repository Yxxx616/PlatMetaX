% MATLAB Code
function [offspring] = updateFunc1513(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Normalize fitness and constraints
    min_fit = min(popfits);
    max_fit = max(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    cv = max(0, cons);
    max_cv = max(cv);
    norm_cv = cv / (max_cv + eps);
    
    % Identify elite solution (best feasible or least infeasible)
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    % Adaptive parameters
    F = 0.5 + 0.3 * (1 - norm_cv);
    sigma = 0.1 * (ub - lb);
    
    % Main mutation operation
    elite_diff = x_elite - popdecs;
    perturbation = sigma .* randn(NP, D) .* (1 + norm_fits(:, ones(1,D)));
    
    mutants = popdecs + F(:, ones(1,D)).*elite_diff + perturbation;
    
    % Boundary handling with bounce-back
    mask_lb = mutants < lb;
    mask_ub = mutants > ub;
    mutants(mask_lb) = lb(mask_lb) + rand(sum(mask_lb(:)),1).*(popdecs(mask_lb) - lb(mask_lb));
    mutants(mask_ub) = ub(mask_ub) - rand(sum(mask_ub(:)),1).*(ub(mask_ub) - popdecs(mask_ub));
    
    % Adaptive crossover
    CR = 0.9 - 0.4 * norm_cv;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Constraint-driven local refinement for top 20%
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.2*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    sigma_refine = 0.05 * (ub - lb) .* (1 - norm_cv(refine_idx, ones(1,D)));
    local_search = popdecs(refine_idx,:) + sigma_refine .* randn(refine_N, D);
    local_search = max(min(local_search, ub), lb);
    
    % Combine with main offspring
    offspring(refine_idx,:) = local_search;
end