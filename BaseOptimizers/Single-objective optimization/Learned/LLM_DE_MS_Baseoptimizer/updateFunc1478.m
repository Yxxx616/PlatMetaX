% MATLAB Code
function [offspring] = updateFunc1478(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Process constraint violations
    cv = max(0, cons);
    norm_cv = cv ./ (max(cv) + eps);
    
    % Find elite, best and worst individuals
    [~, elite_idx] = min(popfits + 1e6*cv);
    [fmin, best_idx] = min(popfits);
    [fmax, ~] = max(popfits);
    x_elite = popdecs(elite_idx, :);
    x_best = popdecs(best_idx, :);
    
    % Generate random indices
    r1 = zeros(NP, 1);
    r2 = zeros(NP, 1);
    for i = 1:NP
        candidates = setdiff(1:NP, [i, elite_idx, best_idx]);
        r1(i) = candidates(randi(length(candidates)));
        candidates = setdiff(candidates, r1(i));
        r2(i) = candidates(randi(length(candidates)));
    end
    
    % Adaptive parameters
    alpha = 0.5 + 0.5 * norm_cv;
    beta = 0.5 - 0.4 * (popfits - fmin) ./ (fmax - fmin + eps);
    F = 0.5 * (1 + cos(pi * (1478/2000))); % Using iteration number
    
    % Mutation
    elite_dir = x_elite - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    best_dir = x_best - popdecs;
    
    mutants = popdecs + F * elite_dir + ...
              alpha(:, ones(1,D)) .* rand_diff + ...
              beta(:, ones(1,D)) .* best_dir;
    
    % Boundary handling
    mutants = min(max(mutants, lb), ub);
    
    % Crossover
    CR = 0.9 * (1 - norm_cv.^0.5);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Local refinement for top 20% solutions
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.2*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    sigma = 0.1 * (1-norm_cv(refine_idx)) .* (ub-lb) .* randn(refine_N,D);
    local_search = popdecs(refine_idx,:) + sigma;
    local_search = min(max(local_search, lb), ub);
    
    % Replace with refined solutions
    offspring(refine_idx,:) = local_search;
end