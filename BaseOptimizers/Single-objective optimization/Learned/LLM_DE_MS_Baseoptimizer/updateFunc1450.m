% MATLAB Code
function [offspring] = updateFunc1450(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    range = ub - lb;
    offspring = zeros(NP, D);
    
    % Process constraints
    cv = max(0, cons); % Constraint violations
    max_cv = max(cv);
    feasible = cv <= 0;
    
    % Compute weighted center
    weights = 1./(1 + cv + eps);
    weights = weights / sum(weights);
    x_center = sum(popdecs .* weights(:), 1);
    
    % Find elite individual
    [~, elite_idx] = min(popfits + 1e6*cv);
    x_elite = popdecs(elite_idx, :);
    
    % Compute adaptive parameters
    f_min = min(popfits);
    f_max = max(popfits);
    F = 0.5 + 0.3 * (popfits - f_min) / (f_max - f_min + eps);
    CR = 0.3 + 0.5 * (1 - cv/(max_cv + eps));
    
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
        
        if feasible(i)
            % Feasible solution: exploit toward elite
            mutants(i,:) = popdecs(i,:) + F(i)*(x_elite - popdecs(i,:)) + ...
                           F(i)*(popdecs(r1,:) - popdecs(r2,:));
        else
            % Infeasible solution: explore around center
            mutants(i,:) = x_center + F(i)*(popdecs(r1,:) - popdecs(r2,:)) + ...
                           0.1*range.*(2*rand(1,D)-1);
        end
    end
    
    % Boundary handling with reflection
    below = mutants < lb;
    above = mutants > ub;
    mutants(below) = 2*lb(below) - mutants(below);
    mutants(above) = 2*ub(above) - mutants(above);
    
    % Crossover
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Local refinement for top feasible solutions
    if any(feasible)
        [~, sorted_idx] = sort(popfits);
        top_idx = sorted_idx(1:min(round(0.2*NP), sum(feasible)));
        sigma = 0.05 * range;
        local_search = popdecs(top_idx,:) + sigma.*randn(length(top_idx),D);
        local_search = min(max(local_search, lb), ub);
        offspring(top_idx,:) = local_search;
    end
end