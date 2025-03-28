% MATLAB Code
function [offspring] = updateFunc1449(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    range = ub - lb;
    offspring = zeros(NP, D);
    
    % Process constraints
    cv = max(0, cons); % Constraint violations
    max_cv = max(cv);
    feasible = cv <= 0;
    
    % Ranking based on fitness and constraints
    [~, fit_rank] = sort(popfits);
    fit_rank = fit_rank / NP; % Normalized rank [0,1]
    
    % Find best individual considering constraints
    [~, best_idx] = min(popfits + 1e6*cv);
    x_best = popdecs(best_idx, :);
    
    % Compute weighted population center
    weights = 1./(1 + cv);
    weights = weights / sum(weights);
    x_center = sum(popdecs .* weights(:), 1);
    
    % Generate random indices (vectorized)
    rand_idx = zeros(NP, 5);
    for i = 1:5
        rand_idx(:,i) = randperm(NP)';
    end
    
    % Ensure distinct indices
    invalid = any(rand_idx == (1:NP)', 2) | any(diff(sort(rand_idx,2),[],2) == 0, 2);
    while any(invalid)
        rand_idx(invalid,:) = randi(NP, sum(invalid), 5);
        invalid = any(rand_idx == (1:NP)', 2) | any(diff(sort(rand_idx,2),[],2) == 0, 2);
    end
    
    % Mutation strategy
    mutants = zeros(NP, D);
    for i = 1:NP
        r1 = rand_idx(i,1);
        r2 = rand_idx(i,2);
        
        if feasible(i) || rand() < 0.7
            % Elite-guided exploitation
            F1 = 0.5 * (1 - cv(i)/(max_cv+eps));
            F2 = 0.7 * (cv(i)/(max_cv+eps));
            mutants(i,:) = popdecs(i,:) + F1*(x_best - popdecs(i,:)) + F2*(popdecs(r1,:) - popdecs(r2,:));
        else
            % Constraint-aware exploration
            F3 = 0.9 * (1 - fit_rank(i));
            F4 = 0.2 * fit_rank(i);
            mutants(i,:) = x_center + F3*(popdecs(r1,:) - popdecs(r2,:)) + F4*range.*randn(1,D);
        end
    end
    
    % Boundary handling with reflection
    below = mutants < lb;
    above = mutants > ub;
    mutants(below) = 2*lb(below) - mutants(below);
    mutants(above) = 2*ub(above) - mutants(above);
    
    % Adaptive crossover rate
    CR = 0.3 + 0.5*(1 - cv/(max_cv+eps));
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Additional local search for top 20% solutions
    if any(feasible)
        [~, sorted_idx] = sort(popfits);
        top_idx = sorted_idx(1:round(0.2*NP));
        sigma = 0.05 * range;
        local_search = popdecs(top_idx,:) + sigma.*randn(length(top_idx),D);
        local_search = min(max(local_search, lb), ub);
        offspring(top_idx,:) = local_search;
    end
end