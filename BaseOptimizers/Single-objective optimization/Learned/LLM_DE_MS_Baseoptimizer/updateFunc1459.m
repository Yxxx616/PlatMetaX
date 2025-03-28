% MATLAB Code
function [offspring] = updateFunc1459(popdecs, popfits, cons)
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
    while any(r1 == r2 | r1 == (1:NP)' | r2 == (1:NP)')
        r1 = randi(NP, NP, 1);
        r2 = randi(NP, NP, 1);
    end
    
    % Adaptive scaling factors
    F_base = 0.5 + 0.3 * rand(NP, 1);
    F_elite = F_base .* (1 - w);
    F_best = F_base .* w;
    F_rand = 0.2 * (1 + rand(NP, 1));
    
    % Mutation vectors
    elite_dir = x_elite - popdecs;
    best_dir = x_best - popdecs;
    rand_dir = popdecs(r1,:) - popdecs(r2,:);
    
    % Combined mutation
    mutants = popdecs + F_elite.*elite_dir + ...
              F_best.*best_dir + ...
              F_rand.*rand_dir;
    
    % Boundary handling with reflection
    below = mutants < lb;
    above = mutants > ub;
    mutants(below) = 2*lb(below) - mutants(below);
    mutants(above) = 2*ub(above) - mutants(above);
    mutants = max(min(mutants, ub), lb);
    
    % Adaptive crossover
    CR = 0.9 - 0.5*w; % More crossover for feasible solutions
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
    
    sigma = 0.1 * (1 - w(top_idx)) .* range;
    local_search = popdecs(top_idx,:) + sigma.*randn(top_N,D);
    local_search = max(min(local_search, ub), lb);
    offspring(top_idx,:) = local_search;
end