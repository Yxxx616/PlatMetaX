% MATLAB Code
function [offspring] = updateFunc1451(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    range = ub - lb;
    offspring = zeros(NP, D);
    
    % Process constraints
    cv = max(0, cons); % Constraint violations
    max_cv = max(cv);
    feasible = cv <= 0;
    
    % Find elite individual (considering both fitness and constraints)
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    % Compute adaptive parameters
    f_min = min(popfits);
    f_max = max(popfits);
    F = 0.5 + 0.3 * (popfits - f_min) / (f_max - f_min + eps);
    CR = 0.3 + 0.5 * (1 - cv/(max_cv + eps));
    alpha = 0.1 * (1 + cv/(max_cv + eps));
    
    % Generate random indices
    rand_idx = zeros(NP, 2);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available), 2));
    end
    
    % Mutation
    mutants = zeros(NP, D);
    for i = 1:NP
        r1 = rand_idx(i,1);
        r2 = rand_idx(i,2);
        
        % Main mutation components
        elite_dir = x_elite - popdecs(i,:);
        diff_vec = popdecs(r1,:) - popdecs(r2,:);
        rand_explore = alpha(i) * randn(1,D) .* range;
        
        % Combine components
        mutants(i,:) = popdecs(i,:) + F(i)*elite_dir + ...
                      (1-cv(i)/(max_cv+eps))*F(i)*diff_vec + rand_explore;
    end
    
    % Boundary handling with bounce-back
    below = mutants < lb;
    above = mutants > ub;
    mutants(below) = lb(below) + 0.5*rand(sum(below(:)),1).*(popdecs(below)-lb(below));
    mutants(above) = ub(above) - 0.5*rand(sum(above(:)),1).*(ub(above)-popdecs(above));
    
    % Crossover
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Enhanced local search for top feasible solutions
    if any(feasible)
        [~, sorted_idx] = sort(popfits);
        top_N = min(5, round(0.1*NP));
        top_idx = sorted_idx(1:top_N);
        
        % Adaptive local search radius
        sigma = 0.05 * (1 + cv(top_idx)/(max_cv+eps)) .* range;
        local_search = popdecs(top_idx,:) + sigma.*randn(length(top_idx),D);
        
        % Ensure boundaries
        local_search = max(min(local_search, ub), lb);
        offspring(top_idx,:) = local_search;
    end
end