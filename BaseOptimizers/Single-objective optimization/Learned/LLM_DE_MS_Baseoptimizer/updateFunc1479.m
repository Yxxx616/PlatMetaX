% MATLAB Code
function [offspring] = updateFunc1479(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Process constraint violations
    cv = max(0, cons);
    norm_cv = cv ./ (max(cv) + eps);
    
    % Identify key individuals
    [~, elite_idx] = min(popfits + 1e6*cv);
    [fmin, best_idx] = min(popfits);
    [fmax, ~] = max(popfits);
    x_elite = popdecs(elite_idx, :);
    x_best = popdecs(best_idx, :);
    
    % Generate random indices for diversity
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(1:NP, [i, elite_idx, best_idx]);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % Adaptive parameters
    alpha = 0.4 + 0.6 * norm_cv;
    beta = 0.5 * (1 - (popfits - fmin) ./ (fmax - fmin + eps));
    F = 0.7 * (1 - norm_cv.^0.5);
    
    % Mutation vectors
    elite_dir = x_elite - popdecs;
    div_dir = popdecs(r(:,1),:) - popdecs(r(:,2),:) + ...
              popdecs(r(:,3),:) - popdecs(r(:,4),:);
    best_dir = x_best - popdecs;
    
    % Combined mutation
    mutants = popdecs + F(:, ones(1,D)) .* elite_dir + ...
              alpha(:, ones(1,D)) .* div_dir + ...
              beta(:, ones(1,D)) .* best_dir;
    
    % Boundary handling
    mutants = min(max(mutants, lb), ub);
    
    % Adaptive crossover
    CR = 0.7 + 0.3 * (1 - sqrt(alpha));
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Local refinement for top solutions
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.25*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    sigma = 0.05 * (1-norm_cv(refine_idx)) .* (ub-lb) .* randn(refine_N,D);
    local_search = popdecs(refine_idx,:) + sigma;
    local_search = min(max(local_search, lb), ub);
    
    offspring(refine_idx,:) = local_search;
end