% MATLAB Code
function [offspring] = updateFunc1515(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Identify elite solution considering constraints
    penalty = 1e6 * max(0, cons);
    [~, elite_idx] = min(popfits + penalty);
    x_elite = popdecs(elite_idx, :);
    
    % Normalize constraints and fitness
    cv = max(0, cons);
    max_cv = max(cv);
    norm_cv = cv / (max_cv + eps);
    
    min_fit = min(popfits);
    max_fit = max(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    % Compute fitness weights for neighborhood
    T = max(1, max_fit - min_fit);
    weights = exp(-popfits/T);
    weights = weights / sum(weights);
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = r1 == r2;
    r2(mask) = mod(r2(mask) + randi(NP-1, sum(mask), 1), NP);
    
    % Adaptive parameters
    F = 0.5 * (1 + norm_cv);
    CR = 0.9 * (1 - norm_cv);
    sigma = 0.2 * (1 - norm_cv) .* (ub - lb);
    
    % Compute neighborhood influence
    neighborhood = zeros(NP, D);
    for i = 1:NP
        for j = 1:NP
            if j ~= i
                neighborhood(i,:) = neighborhood(i,:) + weights(j) * (popdecs(j,:) - popdecs(i,:));
            end
        end
    end
    
    % Mutation operation
    diff_vec = popdecs(r1,:) - popdecs(r2,:);
    mutants = x_elite(ones(NP,1),:) + F(:, ones(1,D)).*diff_vec + sigma(:, ones(1,D)).*neighborhood;
    
    % Boundary handling with bounce-back
    mask_lb = mutants < lb;
    mask_ub = mutants > ub;
    mutants(mask_lb) = lb(mask_lb) + rand(sum(sum(mask_lb)),1).*(ub(mask_lb)-lb(mask_lb));
    mutants(mask_ub) = lb(mask_ub) + rand(sum(sum(mask_ub)),1).*(ub(mask_ub)-lb(mask_ub));
    
    % Crossover
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Local refinement for top 20% solutions
    [~, sorted_idx] = sort(popfits + penalty);
    refine_N = max(1, round(0.2*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    sigma_refine = 0.1 * (1 - norm_cv(refine_idx, ones(1,D))) .* (ub - lb);
    local_search = popdecs(refine_idx,:) + sigma_refine .* randn(refine_N, D);
    local_search = max(min(local_search, ub), lb);
    
    % Combine with main offspring
    offspring(refine_idx,:) = local_search;
end