% MATLAB Code
function [offspring] = updateFunc1448(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    range = ub - lb;
    offspring = zeros(NP, D);
    
    % Process constraints
    cv = max(0, cons); % Constraint violations
    feasible = cv <= 0;
    
    % Combined ranking considering both fitness and constraints
    [~, fit_rank] = sort(popfits);
    [~, cv_rank] = sort(cv);
    R = 0.7 * fit_rank + 0.3 * cv_rank;
    [~, rank_order] = sort(R);
    
    % Divide population into groups
    elite_num = round(0.2*NP);
    weak_num = round(0.2*NP);
    elite_idx = rank_order(1:elite_num);
    weak_idx = rank_order(end-weak_num+1:end);
    middle_idx = setdiff(1:NP, [elite_idx; weak_idx]);
    
    % Find best individual (considering constraints)
    [~, best_idx] = min(popfits + 1e6*cv);
    x_best = popdecs(best_idx, :);
    
    % Compute population center (weighted by inverse constraint violation)
    weights = 1./(1 + cv);
    weights = weights / sum(weights);
    x_center = sum(popdecs .* weights(:), 1);
    
    % Generate all required random indices first
    rand_idx = zeros(NP, 5);
    for i = 1:5
        rand_idx(:,i) = randperm(NP)';
    end
    
    % Ensure distinct indices
    invalid = any(rand_idx == (1:NP)', 2) | any(diff(sort(rand_idx,2),[],2) == 0;
    while any(invalid)
        rand_idx(invalid,:) = randi(NP, sum(invalid), 5);
        invalid = any(rand_idx == (1:NP)', 2) | any(diff(sort(rand_idx,2),[],2) == 0;
    end
    
    % Initialize mutant vectors
    mutants = zeros(NP, D);
    
    % Elite group mutation (local refinement)
    if ~isempty(elite_idx)
        r1 = rand_idx(elite_idx,1);
        r2 = rand_idx(elite_idx,2);
        mutants(elite_idx,:) = popdecs(elite_idx,:) + ...
            0.5*(popdecs(r1,:) - popdecs(r2,:)) + ...
            0.3*(x_best - popdecs(elite_idx,:));
    end
    
    % Middle group mutation (balanced exploration)
    if ~isempty(middle_idx)
        r1 = rand_idx(middle_idx,1);
        r2 = rand_idx(middle_idx,2);
        r3 = rand_idx(middle_idx,3);
        r4 = rand_idx(middle_idx,4);
        mutants(middle_idx,:) = popdecs(r1,:) + ...
            0.7*(popdecs(r2,:) - popdecs(r3,:)) + ...
            0.7*(popdecs(r4,:) - popdecs(rand_idx(middle_idx,5),:));
    end
    
    % Weak group mutation (global exploration)
    if ~isempty(weak_idx)
        r1 = rand_idx(weak_idx,1);
        r2 = rand_idx(weak_idx,2);
        mutants(weak_idx,:) = x_center + ...
            0.7*(popdecs(r1,:) - popdecs(r2,:)) + ...
            0.2*range.*randn(length(weak_idx), D);
    end
    
    % Boundary handling with reflection
    below = mutants < lb;
    above = mutants > ub;
    mutants(below) = 2*lb(below) - mutants(below);
    mutants(above) = 2*ub(above) - mutants(above);
    
    % Adaptive crossover rate
    CR_base = 0.5;
    CR = CR_base + (1-CR_base)*(1 - cv/(max(cv)+eps));
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Additional local search for feasible solutions
    if any(feasible)
        feasible_idx = find(feasible);
        sigma = 0.1 * range;
        local_search = popdecs(feasible_idx,:) + sigma.*randn(length(feasible_idx),D);
        local_search = min(max(local_search, lb), ub);
        % Replace with 50% probability
        replace_mask = rand(length(feasible_idx),1) < 0.5;
        offspring(feasible_idx(replace_mask),:) = local_search(replace_mask,:);
    end
end