% MATLAB Code
function [offspring] = updateFunc1511(popdecs, popfits, cons)
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
    
    % Generate unique random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    invalid = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    while any(invalid)
        r1(invalid) = randi(NP, sum(invalid), 1);
        r2(invalid) = randi(NP, sum(invalid), 1);
        invalid = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    end
    
    % Adaptive parameters
    F1 = 0.8 + 0.1 * norm_fits;
    F2 = 0.5 - 0.4 * norm_cv;
    F3 = 0.2 * (1 - norm_fits);
    sigma = 0.2 * (1 - norm_cv) .* (ub - lb);
    
    % Main mutation operation
    elite_diff = x_elite - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    perturbation = F3 .* randn(NP, D) .* sigma(:, ones(1,D)) .* norm_fits(:, ones(1,D));
    
    mutants = popdecs + F1.*elite_diff + ...
              (F2.*rand(NP,1)).*rand_diff.*(1 - norm_cv(:, ones(1,D)))) + ...
              perturbation;
    
    % Boundary handling with bounce-back
    mask_lb = mutants < lb;
    mask_ub = mutants > ub;
    mutants(mask_lb) = lb(mask_lb) + rand(sum(mask_lb(:)),1).*(popdecs(mask_lb) - lb(mask_lb));
    mutants(mask_ub) = ub(mask_ub) - rand(sum(mask_ub(:)),1).*(ub(mask_ub) - popdecs(mask_ub));
    
    % Adaptive crossover
    CR = 0.9 - 0.5 * norm_cv;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Constraint-driven local refinement
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.2*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    sigma_refine = 0.1 * (ub - lb) .* (1 - norm_cv(refine_idx, ones(1,D)));
    local_search = popdecs(refine_idx,:) + sigma_refine .* randn(refine_N, D);
    local_search = max(min(local_search, ub), lb);
    
    % Combine with main offspring
    offspring(refine_idx,:) = local_search;
end