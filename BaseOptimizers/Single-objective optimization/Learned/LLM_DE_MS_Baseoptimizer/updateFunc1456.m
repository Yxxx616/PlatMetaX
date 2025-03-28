% MATLAB Code
function [offspring] = updateFunc1456(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    range = ub - lb;
    offspring = zeros(NP, D);
    
    % Process constraints
    cv = max(0, cons); % Constraint violations
    w = cv ./ (max(cv) + eps); % Normalized weights [0,1]
    
    % Find elite (best feasible or least infeasible)
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    % Find best feasible if exists
    feasible_mask = cv <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = popdecs(feasible_mask(best_idx), :);
    else
        x_best = x_elite;
    end
    
    % Adaptive parameters
    [~, rank_order] = sort(popfits);
    F = 0.5 + 0.2 * (rank_order/NP); % Better solutions get smaller F
    beta = 0.2 * (1 - w); % More exploration for feasible solutions
    
    % Generate random indices (vectorized)
    [~, rand_idx] = sort(rand(NP, NP), 2);
    rand_idx = rand_idx(:, 2:3); % Two distinct random indices
    
    % Mutation components
    elite_dir = x_elite - popdecs;
    diff_vec = popdecs(rand_idx(:,1), :) - popdecs(rand_idx(:,2), :);
    best_dir = x_best - popdecs;
    rand_explore = beta .* randn(NP, D) .* range;
    
    % Combined mutation
    mutants = popdecs + F.*elite_dir + ...
              ((1-w).*diff_vec + w.*best_dir) + ...
              rand_explore;
    
    % Boundary handling with reflection
    below = mutants < lb;
    above = mutants > ub;
    mutants(below) = 2*lb(below) - mutants(below);
    mutants(above) = 2*ub(above) - mutants(above);
    mutants = max(min(mutants, ub), lb);
    
    % Adaptive crossover
    CR = 0.9 - 0.4*w; % Higher CR for feasible solutions
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Intensification for top 10% solutions
    [~, sorted_idx] = sort(popfits + 1e6*cv);
    top_N = max(1, round(0.1*NP));
    top_idx = sorted_idx(1:top_N);
    
    % Local search with adaptive radius
    sigma = 0.1 * (1 + w(top_idx)) .* range;
    local_search = popdecs(top_idx,:) + sigma.*randn(top_N,D);
    local_search = max(min(local_search, ub), lb);
    offspring(top_idx,:) = local_search;
end