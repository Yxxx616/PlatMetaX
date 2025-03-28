% MATLAB Code
function [offspring] = updateFunc1471(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Process constraint violations
    cv = max(0, cons);
    w = cv ./ (max(cv) + eps);
    
    % Select elite individual (considering both fitness and constraints)
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    % Select best feasible individual
    feasible_mask = cv <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = popdecs(feasible_mask,:);
        x_best = x_best(best_idx, :);
    else
        x_best = x_elite;
    end
    
    % Generate random indices (distinct from current and each other)
    r = zeros(NP, 3);
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        r(i,:) = candidates(randperm(length(candidates), 3));
    end
    
    % Calculate direction vectors
    elite_dir = x_elite - popdecs;
    best_dir = x_best - popdecs;
    opposition = (lb + ub - popdecs(r(:,1),:) - popdecs);
    rand_diff = popdecs(r(:,2),:) - popdecs(r(:,3),:);
    
    % Adaptive scaling factors
    F1 = 0.5 + 0.2*(1 - w);
    F2 = 0.3 + 0.2*w;
    F3 = 0.2 + 0.1*rand(NP,1);
    F4 = 0.1 + 0.1*rand(NP,1);
    
    % Composite mutation
    mutants = popdecs + F1.*elite_dir + F2.*best_dir + ...
              F3.*opposition + F4.*rand_diff;
    
    % Boundary handling with reflection
    below = mutants < lb;
    above = mutants > ub;
    mutants(below) = 2*lb(below) - mutants(below);
    mutants(above) = 2*ub(above) - mutants(above);
    mutants = max(min(mutants, ub), lb);
    
    % Adaptive crossover
    CR = 0.9 * (1 - w.^0.5);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Constraint-driven local refinement for top 20% solutions
    [~, sorted_idx] = sort(cv);
    refine_N = max(1, round(0.2*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    sigma = 0.1 * (1-w(refine_idx)) .* (ub-lb) .* (1 + 0.1*randn(refine_N,D));
    local_search = popdecs(refine_idx,:) + sigma;
    local_search = max(min(local_search, ub), lb);
    
    % Replace with refined solutions
    offspring(refine_idx,:) = local_search;
end