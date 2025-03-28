% MATLAB Code
function [offspring] = updateFunc1508(popdecs, popfits, cons)
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
    
    % Identify elite solution (considering both fitness and constraints)
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    % Generate random indices for differential vectors
    r1 = zeros(NP, 1);
    r2 = zeros(NP, 1);
    for i = 1:NP
        available = setdiff(1:NP, i);
        r1(i) = available(randi(length(available)));
        available = setdiff(available, r1(i));
        r2(i) = available(randi(length(available)));
    end
    
    % Adaptive scaling factors
    F1 = 0.7 + 0.2 * norm_fits;
    F2 = 0.5 - 0.3 * norm_fits;
    F3 = 0.1 * norm_cv;
    
    % Main mutation operation
    elite_diff = x_elite - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    perturbation = F3 .* randn(NP, D) .* norm_cv;
    
    mutants = popdecs + F1.*elite_diff + F2.*rand_diff.*(1 - norm_fits) + perturbation;
    
    % Boundary handling with reflection
    mask_lb = mutants < lb;
    mask_ub = mutants > ub;
    mutants(mask_lb) = 2*lb(mask_lb) - mutants(mask_lb);
    mutants(mask_ub) = 2*ub(mask_ub) - mutants(mask_ub);
    
    % Adaptive crossover
    CR = 0.9 - 0.2 * norm_cv;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Constraint-aware local search for top 30%
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.3*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    % Adaptive step size based on constraints
    sigma = 0.1 * (ub - lb) .* (1 - norm_cv(refine_idx));
    local_search = popdecs(refine_idx,:) + sigma .* randn(refine_N, D);
    
    % Project back to bounds
    local_search = max(min(local_search, ub), lb);
    offspring(refine_idx,:) = local_search;
end