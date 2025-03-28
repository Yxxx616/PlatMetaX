% MATLAB Code
function [offspring] = updateFunc1455(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    range = ub - lb;
    offspring = zeros(NP, D);
    
    % Process constraints
    cv = max(0, cons); % Constraint violations
    w = cv ./ (max(cv) + eps); % Normalized constraint violation weights
    
    % Find elite individual considering both fitness and constraints
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    % Compute adaptive parameters
    f_min = min(popfits);
    f_max = max(popfits);
    F = 0.5 + 0.3 * (popfits - f_min) ./ (f_max - f_min + eps);
    beta = 0.3; % Exploration factor
    
    % Generate random indices without loops
    [~, rand_idx] = sort(rand(NP, NP), 2);
    rand_idx = rand_idx(:, 2:3); % Get two distinct random indices
    
    % Vectorized mutation
    elite_dir = x_elite - popdecs;
    diff_vec = popdecs(rand_idx(:,1), :) - popdecs(rand_idx(:,2), :);
    rand_explore = beta .* w .* randn(NP, D) .* range;
    
    mutants = popdecs + F.*elite_dir + (0.5 + 0.5*w).*diff_vec + rand_explore;
    
    % Boundary handling with bounce-back
    below = mutants < lb;
    above = mutants > ub;
    mutants(below) = lb(below) + 0.5*rand(sum(below(:)),1).*(popdecs(below)-lb(below));
    mutants(above) = ub(above) - 0.5*rand(sum(above(:)),1).*(ub(above)-popdecs(above));
    
    % Adaptive crossover
    CR = 0.9 - 0.5*w; % More crossover for feasible solutions
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Intensification for top feasible solutions
    feasible = cv <= 0;
    if any(feasible)
        [~, sorted_idx] = sort(popfits);
        top_N = min(5, sum(feasible));
        top_idx = sorted_idx(1:top_N);
        
        % Local search with adaptive radius
        sigma = 0.05 * (1 + w(top_idx)) .* range;
        local_search = popdecs(top_idx,:) + sigma.*randn(top_N,D);
        local_search = max(min(local_search, ub), lb);
        offspring(top_idx,:) = local_search;
    end
end