% MATLAB Code
function [offspring] = updateFunc1488(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Normalize constraints and fitness
    cv = max(0, cons);
    norm_cv = cv ./ (max(cv) + eps);
    fmin = min(popfits);
    fmax = max(popfits);
    norm_fit = (popfits - fmin) ./ (fmax - fmin + eps);
    
    % Identify key individuals
    [~, elite_idx] = min(popfits + 1e6*cv);
    feasible_mask = (cons <= 0);
    if any(feasible_mask)
        [~, best_feas_idx] = min(popfits .* feasible_mask);
    else
        [~, best_feas_idx] = min(popfits);
    end
    x_elite = popdecs(elite_idx, :);
    x_best = popdecs(best_feas_idx, :);
    
    % Generate random indices ensuring they're distinct
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = (r1 == r2) | (r1 == (1:NP)') | (r2 == (1:NP)');
    while any(mask)
        r1(mask) = randi(NP, sum(mask), 1);
        r2(mask) = randi(NP, sum(mask), 1);
        mask = (r1 == r2) | (r1 == (1:NP)') | (r2 == (1:NP)');
    end
    
    % Adaptive parameters
    F = 0.5 + 0.3 * (1 - norm_fit) .* (1 - norm_cv);
    alpha = 0.2 * norm_cv;
    beta = 0.1 * norm_fit;
    
    % Mutation components
    elite_dir = x_elite - popdecs;
    best_dir = x_best - popdecs;
    
    % Constraint-aware diversity direction
    cv_weight = 1 + alpha .* (cv(r1) - cv(r2));
    div_dir = (popdecs(r1,:) - popdecs(r2,:)) .* cv_weight(:, ones(1, D));
    
    % Combined mutation
    mutants = popdecs + F .* elite_dir + alpha .* best_dir + beta .* div_dir;
    
    % Boundary handling
    mutants = min(max(mutants, lb), ub);
    
    % Adaptive crossover
    CR = 0.9 - 0.4 * norm_cv;
    mask = rand(NP,D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Local refinement for top solutions
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.1*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    sigma_refine = 0.02 * (1-norm_cv(refine_idx)) .* (ub-lb) .* randn(refine_N,D);
    local_search = popdecs(refine_idx,:) + sigma_refine;
    local_search = min(max(local_search, lb), ub);
    
    offspring(refine_idx,:) = local_search;
end