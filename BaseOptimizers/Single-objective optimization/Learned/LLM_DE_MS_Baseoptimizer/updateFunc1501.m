% MATLAB Code
function [offspring] = updateFunc1501(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Normalize fitness and constraints
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    cv = max(0, cons);
    norm_cv = cv / (max(cv) + eps);
    
    % Identify elite solution (considering constraints)
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
    F1 = 0.5 + 0.3 * norm_fits;
    F2 = 0.3 + 0.2 * (1 - norm_cv);
    F3 = 0.2 * norm_cv;
    
    % Main mutation operation
    elite_diff = x_elite - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    perturbation = F3 .* randn(NP, D) .* norm_fits;
    
    mutants = popdecs + F1.*elite_diff + F2.*rand_diff.*(1 - norm_cv) + perturbation;
    
    % Boundary handling with reflection
    mask_lb = mutants < lb;
    mask_ub = mutants > ub;
    mutants(mask_lb) = 2*lb(mask_lb) - mutants(mask_lb);
    mutants(mask_ub) = 2*ub(mask_ub) - mutants(mask_ub);
    
    % Adaptive crossover
    CR = 0.9 * (1 - norm_cv) + 0.1;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Local refinement for top 20% solutions
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.2*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    % Local search with adaptive step size
    sigma = 0.1 * (ub - lb) .* (1 - norm_fits(refine_idx));
    local_search = popdecs(refine_idx,:) + sigma .* randn(refine_N, D);
    local_search = min(max(local_search, lb), ub);
    offspring(refine_idx,:) = local_search;
end