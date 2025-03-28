% MATLAB Code
function [offspring] = updateFunc1492(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Normalize fitness and constraints
    fmin = min(popfits);
    fmax = max(popfits);
    norm_fit = (popfits - fmin) ./ (fmax - fmin + eps);
    
    cv = max(0, cons);
    cv_max = max(cv);
    norm_cv = cv ./ (cv_max + eps);
    
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
    
    % Generate unique random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = (r1 == r2) | (r1 == (1:NP)') | (r2 == (1:NP)');
    while any(mask)
        r1(mask) = randi(NP, sum(mask), 1);
        r2(mask) = randi(NP, sum(mask), 1);
        mask = (r1 == r2) | (r1 == (1:NP)') | (r2 == (1:NP)');
    end
    
    % Adaptive parameters
    w = (1 - norm_fit) .* (1 + norm_cv);
    alpha = 0.5*(1 - norm_fit) + 0.1*norm_cv;
    beta = 0.3;
    CR = 0.9 - 0.4*norm_cv;
    
    % Base vector with adaptive blending
    x_base = x_elite + beta * (x_best - x_elite);
    
    % Weighted difference vectors
    diff_vec = popdecs(r1,:) - popdecs(r2,:);
    weighted_diff = diff_vec .* w(:, ones(1, D));
    
    % Mutation
    mutants = x_base + alpha(:, ones(1, D)) .* weighted_diff;
    
    % Boundary handling
    mutants = min(max(mutants, lb), ub);
    
    % Crossover
    mask = rand(NP,D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Local refinement for top solutions
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.2*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    sigma = 0.05 * (1 - norm_cv(refine_idx)) .* (ub-lb) .* randn(refine_N,D);
    local_search = popdecs(refine_idx,:) + sigma;
    local_search = min(max(local_search, lb), ub);
    
    offspring(refine_idx,:) = local_search;
end