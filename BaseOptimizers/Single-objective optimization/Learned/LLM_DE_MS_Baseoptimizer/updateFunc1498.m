% MATLAB Code
function [offspring] = updateFunc1498(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Calculate adaptive weights
    mean_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    cv = max(0, cons);
    max_cv = max(cv) + eps;
    
    % Combined weight considering both fitness and constraints
    w = 1 ./ (1 + exp(-(popfits - mean_fit)./std_fit)) .* ...
        1 ./ (1 + exp(cv./max_cv));
    
    % Identify elite solution (considering both fitness and constraints)
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    % Identify best feasible solution
    feasible = (cons <= 0);
    if any(feasible)
        [~, best_feas_idx] = min(popfits .* feasible);
        x_best = popdecs(best_feas_idx, :);
    else
        x_best = x_elite;
    end
    
    % Generate random indices (2 distinct vectors)
    r = zeros(NP, 2);
    for i = 1:2
        r(:,i) = randperm(NP)';
    end
    
    % Ensure all indices are distinct and not current
    mask = any(r == (1:NP)', 2) | (r(:,1) == r(:,2));
    while any(mask)
        r(mask,:) = randi(NP, sum(mask), 2);
        mask = any(r == (1:NP)', 2) | (r(:,1) == r(:,2));
    end
    
    % Adaptive parameters with stronger constraint influence
    F1 = 0.5 + 0.2*w - 0.3*(cv./max_cv);
    F2 = 0.3 + 0.1*w;
    
    % Enhanced mutation with elite and best guidance
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = x_best - popdecs;
    mutants = x_elite + F1.*diff1 + F2.*diff2;
    
    % Boundary handling with reflection
    mask_lb = mutants < lb;
    mask_ub = mutants > ub;
    mutants(mask_lb) = 2*lb(mask_lb) - mutants(mask_lb);
    mutants(mask_ub) = 2*ub(mask_ub) - mutants(mask_ub);
    mutants = min(max(mutants, lb), ub);
    
    % Crossover with adaptive CR
    CR = 0.9 - 0.5*(cv./max_cv);
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Local refinement for top solutions (30% of population)
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.3*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    % Adaptive local search with weight-based step size
    sigma = 0.1 * (1-w(refine_idx)) .* (ub-lb) .* randn(refine_N,D);
    local_search = popdecs(refine_idx,:) + sigma;
    
    % Boundary handling for local search
    local_search = min(max(local_search, lb), ub);
    offspring(refine_idx,:) = local_search;
end