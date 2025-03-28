% MATLAB Code
function [offspring] = updateFunc1510(popdecs, popfits, cons)
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
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    invalid = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    while any(invalid)
        r1(invalid) = randi(NP, sum(invalid), 1);
        r2(invalid) = randi(NP, sum(invalid), 1);
        invalid = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    end
    
    % Adaptive parameters
    F1 = 0.7 + 0.2 * norm_fits;
    F2 = 0.5 - 0.3 * norm_cv;
    F3 = 0.1 * (1 - norm_fits);
    xi = rand(NP, 1);
    sigma = 0.1 * (1 - norm_cv) .* (ub - lb);
    
    % Main mutation operation
    elite_diff = x_elite - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    perturbation = F3 .* randn(NP, D) .* sigma(:, ones(1,D)) .* norm_fits(:, ones(1,D));
    
    mutants = popdecs + F1.*elite_diff + ...
              (F2.*xi).*rand_diff.*(1 - norm_cv(:, ones(1,D))) + ...
              perturbation;
    
    % Boundary handling with reflection
    mask_lb = mutants < lb;
    mask_ub = mutants > ub;
    mutants(mask_lb) = 2*lb(mask_lb) - mutants(mask_lb);
    mutants(mask_ub) = 2*ub(mask_ub) - mutants(mask_ub);
    
    % Adaptive crossover
    CR = 0.9 - 0.4 * norm_cv;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Constraint-aware local refinement for top solutions
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.1*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    sigma_refine = 0.05 * (ub - lb) .* (1 - norm_cv(refine_idx, ones(1,D)));
    local_search = popdecs(refine_idx,:) + sigma_refine .* randn(refine_N, D);
    local_search = max(min(local_search, ub), lb);
    
    % Combine with main offspring
    offspring(refine_idx,:) = local_search;
end