% MATLAB Code
function [offspring] = updateFunc1500(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Normalize fitness and constraints
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    cv = max(0, cons);
    max_cv = max(cv) + eps;
    norm_cv = cv / max_cv;
    
    % Adaptive weights combining fitness and constraints
    w = 0.6 * (1 - norm_fits) + 0.4 * (1 - norm_cv);
    
    % Identify elite and best feasible solutions
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    feasible = (cons <= 0);
    if any(feasible)
        [~, best_feas_idx] = min(popfits .* feasible);
        x_best = popdecs(best_feas_idx, :);
    else
        x_best = x_elite;
    end
    
    % Generate random indices for differential vectors
    r = zeros(NP, 2);
    for i = 1:NP
        available = setdiff(1:NP, i);
        r(i,:) = available(randperm(length(available), 2));
    end
    
    % Adaptive scaling factors
    F1 = 0.5 * w + 0.3 * (1 - norm_cv);
    F2 = 0.3 * (1 - w) + 0.2 * norm_cv;
    F3 = 0.2 * norm_cv;
    
    % Main mutation operation
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = x_best - popdecs;
    perturbation = F3 .* randn(NP, D) .* (ub - lb);
    mutants = x_elite + F1.*diff1 + F2.*diff2 + perturbation;
    
    % Boundary handling with reflection
    mask_lb = mutants < lb;
    mask_ub = mutants > ub;
    mutants(mask_lb) = 2*lb(mask_lb) - mutants(mask_lb);
    mutants(mask_ub) = 2*ub(mask_ub) - mutants(mask_ub);
    
    % Adaptive crossover
    CR = 0.85 * w + 0.15;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Local refinement for top solutions
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.25*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    % Diversity-based local search
    pop_std = std(popdecs) + eps;
    sigma = 0.1 * pop_std .* randn(refine_N, D);
    local_search = popdecs(refine_idx,:) + sigma;
    local_search = min(max(local_search, lb), ub);
    offspring(refine_idx,:) = local_search;
end