% MATLAB Code
function [offspring] = updateFunc1460(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    range = ub - lb;
    offspring = zeros(NP, D);
    
    % Process constraints
    cv = max(0, cons); % Constraint violations
    w = cv ./ (max(cv) + eps); % Normalized weights [0,1]
    
    % Find elite individual (considering both fitness and constraints)
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    % Find best feasible individual
    feasible = cv <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible,:);
        x_best = x_best(best_idx, :);
    else
        x_best = x_elite;
    end
    
    % Generate random indices for differential vectors
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = (r1 == r2) | (r1 == (1:NP)') | (r2 == (1:NP)');
    while any(mask)
        r1(mask) = randi(NP, sum(mask), 1);
        r2(mask) = randi(NP, sum(mask), 1);
        mask = (r1 == r2) | (r1 == (1:NP)') | (r2 == (1:NP)');
    end
    
    % Adaptive scaling factors with small noise
    noise = 0.1 * randn(NP, 1);
    F_elite = 0.6 * (1 - w) .* (1 + noise);
    F_best = 0.4 * w .* (1 + noise);
    F_rand = 0.2 * (1 + noise);
    
    % Mutation vectors
    elite_dir = x_elite - popdecs;
    best_dir = x_best - popdecs;
    rand_dir = popdecs(r1,:) - popdecs(r2,:);
    
    % Combined mutation
    mutants = popdecs + F_elite.*elite_dir + F_best.*best_dir + F_rand.*rand_dir;
    
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
    
    % Local refinement for top 20% solutions
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    top_N = max(1, round(0.2*NP));
    top_idx = sorted_idx(1:top_N);
    
    sigma = 0.05 * (1 - w(top_idx)) .* range;
    local_search = popdecs(top_idx,:) + sigma.*randn(top_N,D);
    local_search = max(min(local_search, ub), lb);
    offspring(top_idx,:) = local_search;
end