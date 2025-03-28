% MATLAB Code
function [offspring] = updateFunc1499(popdecs, popfits, cons)
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
    
    % Enhanced adaptive parameters
    F1 = 0.6 * w + 0.2 * (1 - cv./max_cv);
    F2 = 0.4 * (1 - w) + 0.1 * (cv./max_cv);
    
    % Directional mutation with elite and best guidance
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = x_best - popdecs;
    mutants = x_elite + F1.*diff1 + F2.*diff2;
    
    % Boundary handling with reflection and clamping
    mask_lb = mutants < lb;
    mask_ub = mutants > ub;
    mutants(mask_lb) = min(2*lb(mask_lb) - mutants(mask_lb), ub(mask_lb));
    mutants(mask_ub) = max(2*ub(mask_ub) - mutants(mask_ub), lb(mask_ub));
    
    % Adaptive crossover
    CR = 0.9 * w + 0.1 * (1 - cv./max_cv);
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Local refinement for top solutions (20% of population)
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.2*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    % Weight-based local search
    sigma = 0.05 * (1-w(refine_idx)) .* (ub-lb) .* randn(refine_N,D);
    local_search = popdecs(refine_idx,:) + sigma;
    
    % Boundary handling for local search
    local_search = min(max(local_search, lb), ub);
    offspring(refine_idx,:) = local_search;
end