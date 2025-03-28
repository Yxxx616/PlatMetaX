% MATLAB Code
function [offspring] = updateFunc1468(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    range = ub - lb;
    offspring = zeros(NP, D);
    
    % Process constraints
    cv = max(0, cons);
    w = cv ./ (max(cv) + eps);
    
    % Select elite and best feasible
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    feasible = cv <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible,:);
        x_best = x_best(best_idx, :);
    else
        x_best = x_elite;
    end
    
    % Generate 5 distinct random indices
    r = zeros(NP, 5);
    for i = 1:5
        r(:,i) = randi(NP, NP, 1);
    end
    
    % Ensure distinct indices from current and each other
    for i = 1:NP
        while any(r(i,:) == i) || numel(unique(r(i,:))) < 5
            r(i,:) = randi(NP, 1, 5);
        end
    end
    
    % Adaptive scaling factors
    F1 = 0.5*(1-w) + 0.1*randn(NP,1);
    F2 = 0.3*w + 0.1*randn(NP,1);
    F3 = 0.2*(1-w.^2) + 0.1*randn(NP,1);
    F4 = 0.1*w.^2 + 0.1*randn(NP,1);
    
    % Composite mutation with two difference vectors
    elite_diff = x_elite - popdecs;
    best_diff = x_best - popdecs(r(:,1),:);
    rand_diff1 = popdecs(r(:,2),:) - popdecs(r(:,3),:);
    rand_diff2 = popdecs(r(:,4),:) - popdecs(r(:,5),:);
    
    mutants = popdecs + F1.*elite_diff + F2.*best_diff + ...
              F3.*rand_diff1 + F4.*rand_diff2;
    
    % Boundary handling with bounce-back
    below = mutants < lb;
    above = mutants > ub;
    mutants(below) = lb(below) + 0.5*rand(sum(below(:)),1).*(popdecs(below)-lb(below));
    mutants(above) = ub(above) - 0.5*rand(sum(above(:)),1).*(ub(above)-popdecs(above));
    
    % Adaptive crossover with constraint-awareness
    CR = 0.85 - 0.4*w;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Local refinement for top 30% solutions
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    top_N = max(1, round(0.3*NP));
    top_idx = sorted_idx(1:top_N);
    
    % Adaptive local search radius
    sigma = 0.05*(1-w(top_idx)).*range .* (1 + randn(top_N,D));
    local_search = popdecs(top_idx,:) + sigma;
    local_search = max(min(local_search, ub), lb);
    
    % Blend with original population (50% chance)
    blend_mask = rand(top_N,1) < 0.5;
    offspring(top_idx(blend_mask),:) = 0.5*popdecs(top_idx(blend_mask),:) + ...
                                     0.5*local_search(blend_mask,:);
    offspring(top_idx(~blend_mask),:) = local_search(~blend_mask,:);
end