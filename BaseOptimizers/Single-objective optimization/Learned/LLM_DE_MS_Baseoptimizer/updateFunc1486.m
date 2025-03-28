% MATLAB Code
function [offspring] = updateFunc1486(popdecs, popfits, cons)
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
    
    % Generate random indices for diversity component
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(1:NP, [i, elite_idx, best_feas_idx]);
        if length(candidates) >= 4
            r(i,:) = candidates(randperm(length(candidates), 4));
        else
            r(i,:) = randperm(NP, 4);
        end
    end
    
    % Adaptive parameters
    F = 0.6 + 0.2 * (1 - norm_cv) .* (1 - norm_fit);
    alpha = 0.3 * (1 - norm_fit) .* (1 - norm_cv);
    beta = 0.2 * norm_cv;
    gamma = 0.1 * norm_fit;
    sigma = 0.1 * (1 - norm_cv) * (ub - lb);
    
    % Mutation components
    elite_dir = x_elite - popdecs;
    best_dir = x_best - popdecs;
    div_dir = popdecs(r(:,1),:) - popdecs(r(:,2),:) + ...
              popdecs(r(:,3),:) - popdecs(r(:,4),:);
    perturbation = gamma .* sigma .* randn(NP, D);
    
    % Combined mutation
    mutants = popdecs + F .* elite_dir + ...
              alpha .* best_dir + ...
              beta .* div_dir + perturbation;
    
    % Boundary handling
    mutants = min(max(mutants, lb), ub);
    
    % Adaptive crossover
    CR = 0.85 - 0.35 * norm_cv;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Local refinement for top solutions
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.15*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    sigma_refine = 0.03 * (1-norm_cv(refine_idx)) .* (ub-lb) .* randn(refine_N,D);
    local_search = popdecs(refine_idx,:) + sigma_refine;
    local_search = min(max(local_search, lb), ub);
    
    offspring(refine_idx,:) = local_search;
end