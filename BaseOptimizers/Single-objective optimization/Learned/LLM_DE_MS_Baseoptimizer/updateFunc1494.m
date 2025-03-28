% MATLAB Code
function [offspring] = updateFunc1494(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Calculate adaptive weights
    mean_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    cv = max(0, cons);
    max_cv = max(cv) + eps;
    
    w = 1 ./ (1 + exp(-(popfits - mean_fit)./std_fit)) .* ...
        1 ./ (1 + exp(cv./max_cv));
    
    % Identify elite and best feasible
    [~, elite_idx] = min(popfits + 1e6*cv);
    feasible = (cons <= 0);
    if any(feasible)
        [~, best_feas_idx] = min(popfits .* feasible);
    else
        [~, best_feas_idx] = min(popfits);
    end
    
    % Generate random indices (4 distinct vectors)
    r = zeros(NP, 4);
    for i = 1:4
        r(:,i) = randperm(NP)';
    end
    
    % Ensure all indices are distinct and not current
    mask = any(r == (1:NP)', 2) | ...
           (r(:,1) == r(:,2)) | (r(:,1) == r(:,3)) | (r(:,1) == r(:,4)) | ...
           (r(:,2) == r(:,3)) | (r(:,2) == r(:,4)) | (r(:,3) == r(:,4));
    while any(mask)
        r(mask,:) = randi(NP, sum(mask), 4);
        mask = any(r == (1:NP)', 2) | ...
               (r(:,1) == r(:,2)) | (r(:,1) == r(:,3)) | (r(:,1) == r(:,4)) | ...
               (r(:,2) == r(:,3)) | (r(:,2) == r(:,4)) | (r(:,3) == r(:,4));
    end
    
    % Adaptive parameters
    F = 0.5 + 0.3*w - 0.2*(cv./max_cv);
    CR = 0.9 - 0.5*(cv./max_cv);
    
    % Elite-guided mutation
    x_elite = popdecs(elite_idx, :);
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    mutants = x_elite + F.*(diff1 + diff2);
    
    % Boundary handling
    mutants = min(max(mutants, lb), ub);
    
    % Crossover
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Local refinement for top solutions
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    refine_N = max(1, round(0.2*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    sigma = 0.1 * (1-w(refine_idx)) .* (ub-lb) .* randn(refine_N,D);
    local_search = popdecs(refine_idx,:) + sigma;
    local_search = min(max(local_search, lb), ub);
    
    offspring(refine_idx,:) = local_search;
end