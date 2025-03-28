% MATLAB Code
function [offspring] = updateFunc1470(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Constraint violation processing
    cv = max(0, cons);
    w = cv ./ (max(cv) + eps);
    
    % Select elite individual (considering constraints)
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    % Select best feasible individual
    feasible = cv <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible,:);
        x_best = x_best(best_idx, :);
    else
        x_best = x_elite;
    end
    
    % Generate random indices (distinct from current and each other)
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % Direction vectors
    elite_diff = x_elite - popdecs;
    best_diff = x_best - popdecs(r(:,1),:);
    opposition = (lb + ub - popdecs(r(:,2),:) - popdecs);
    rand_diff = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    
    % Adaptive scaling factors
    F = 0.5 + 0.3*cos(pi*w) + 0.2*randn(NP,1);
    
    % Composite mutation
    mutants = popdecs + F.*elite_diff + 0.5*F.*best_diff + ...
              0.3*F.*opposition + 0.2*F.*rand_diff;
    
    % Boundary handling with reflection
    below = mutants < lb;
    above = mutants > ub;
    mutants(below) = 2*lb(below) - mutants(below);
    mutants(above) = 2*ub(above) - mutants(above);
    mutants = max(min(mutants, ub), lb);
    
    % Adaptive crossover
    CR = 0.9 * (1 - sqrt(w));
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Constraint-driven local refinement
    [~, sorted_idx] = sort(cv);
    refine_N = max(1, round(0.2*NP));
    refine_idx = sorted_idx(1:refine_N);
    
    sigma = 0.1 * (1-w(refine_idx)) .* (ub-lb) .* (1 + 0.1*randn(refine_N,D));
    local_search = popdecs(refine_idx,:) + sigma;
    local_search = max(min(local_search, ub), lb);
    
    % Replace with refined solutions
    offspring(refine_idx,:) = local_search;
end