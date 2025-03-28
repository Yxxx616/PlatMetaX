% MATLAB Code
function [offspring] = updateFunc1463(popdecs, popfits, cons)
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
    
    % Generate distinct random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    r3 = randi(NP, NP, 1);
    
    % Ensure distinct indices
    mask = (r1 == r2) | (r1 == r3) | (r2 == r3) | (r1 == (1:NP)') | (r2 == (1:NP)') | (r3 == (1:NP)');
    while any(mask)
        r1(mask) = randi(NP, sum(mask), 1);
        r2(mask) = randi(NP, sum(mask), 1);
        r3(mask) = randi(NP, sum(mask), 1);
        mask = (r1 == r2) | (r1 == r3) | (r2 == r3) | (r1 == (1:NP)') | (r2 == (1:NP)') | (r3 == (1:NP)');
    end
    
    % Adaptive scaling factors
    F1 = 0.5*(1-w) + 0.1*randn(NP,1);
    F2 = 0.3*(1-w) + 0.1*randn(NP,1);
    F3 = 0.2 + 0.1*randn(NP,1);
    
    % Composite mutation
    elite_diff = x_elite - popdecs;
    best_diff = x_best - popdecs(r1,:);
    rand_diff = popdecs(r2,:) - popdecs(r3,:);
    mutants = popdecs + F1.*elite_diff + F2.*best_diff + F3.*rand_diff;
    
    % Boundary handling with reflection
    below = mutants < lb;
    above = mutants > ub;
    mutants(below) = 2*lb(below) - mutants(below);
    mutants(above) = 2*ub(above) - mutants(above);
    mutants = max(min(mutants, ub), lb);
    
    % Adaptive crossover
    CR = 0.9 - 0.5*w;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Local refinement for top solutions
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    top_N = max(1, round(0.1*NP));
    top_idx = sorted_idx(1:top_N);
    
    sigma = 0.1*(1-w(top_idx)).*range;
    local_search = popdecs(top_idx,:) + sigma.*randn(top_N,D);
    local_search = max(min(local_search, ub), lb);
    offspring(top_idx,:) = local_search;
end