% MATLAB Code
function [offspring] = updateFunc1514(popdecs, popfits, cons)
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
    
    % Generate random indices for differential vectors
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2 = randi(NP, NP, 1);
    end
    
    % Adaptive parameters
    F = 0.4 + 0.4 * (1 - norm_cv);
    sigma = 0.2 * (ub - lb);
    CR = 0.85 - 0.35 * norm_cv;
    
    % Main mutation operation
    diff_vec = popdecs(r1,:) - popdecs(r2,:);
    perturbation = sigma .* randn(NP, D) .* (1 + norm_fits(:, ones(1,D)));
    
    mutants = x_elite(ones(NP,1),:) + F(:, ones(1,D)).*diff_vec + perturbation;
    
    % Boundary handling with reflection
    mask_lb = mutants < lb;
    mask_ub = mutants > ub;
    mutants(mask_lb) = 2*lb(mask_lb) - mutants(mask_lb);
    mutants(mask_ub) = 2*ub(mask_ub) - mutants(mask_ub);
    mutants = max(min(mutants, ub), lb);
    
    % Fitness-aware crossover
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Constraint-driven local refinement for top solutions
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.3*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    sigma_refine = 0.1 * (ub - lb) .* (1 - norm_cv(refine_idx, ones(1,D)));
    local_search = popdecs(refine_idx,:) + sigma_refine .* randn(refine_N, D);
    local_search = max(min(local_search, ub), lb);
    
    % Combine with main offspring
    offspring(refine_idx,:) = local_search;
end