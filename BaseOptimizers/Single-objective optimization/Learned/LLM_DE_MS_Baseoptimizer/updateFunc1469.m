% MATLAB Code
function [offspring] = updateFunc1469(popdecs, popfits, cons)
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
    
    % Generate 4 distinct random indices
    r = zeros(NP, 4);
    for i = 1:4
        r(:,i) = randi(NP, NP, 1);
    end
    
    % Ensure distinct indices from current and each other
    for i = 1:NP
        while any(r(i,:) == i) || numel(unique(r(i,:))) < 4
            r(i,:) = randi(NP, 1, 4);
        end
    end
    
    % Adaptive scaling factors
    F1 = 0.6*(1-w) + 0.1*randn(NP,1);
    F2 = 0.4*w + 0.1*randn(NP,1);
    F3 = 0.2*(1-w.^2) + 0.1*randn(NP,1);
    F4 = 0.1*w.^2 + 0.1*randn(NP,1);
    
    % Composite mutation
    elite_diff = x_elite - popdecs;
    best_diff = x_best - popdecs(r(:,1),:);
    opposition = (lb + ub - popdecs(r(:,2),:) - popdecs);
    rand_diff = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    
    mutants = popdecs + F1.*elite_diff + F2.*best_diff + ...
              F3.*opposition + F4.*rand_diff;
    
    % Boundary handling with reflection
    below = mutants < lb;
    above = mutants > ub;
    mutants(below) = 2*lb(below) - mutants(below);
    mutants(above) = 2*ub(above) - mutants(above);
    mutants = max(min(mutants, ub), lb);
    
    % Adaptive crossover with constraint-awareness
    CR = 0.9 - 0.5*w;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Constraint-driven local search
    [~, sorted_idx] = sort(cv);
    top_N = max(1, round(0.3*NP));
    top_idx = sorted_idx(1:top_N);
    
    % Adaptive local search radius
    sigma = 0.1*(1-w(top_idx)).*range .* (1 + 0.1*randn(top_N,D));
    local_search = popdecs(top_idx,:) + sigma;
    local_search = max(min(local_search, ub), lb);
    
    % Replace worst solutions with refined ones
    offspring(top_idx,:) = local_search;
end