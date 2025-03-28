% MATLAB Code
function [offspring] = updateFunc1343(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Normalize constraint violations
    cv_abs = abs(cons);
    cv_max = max(cv_abs) + 1e-12;
    cv_norm = cv_abs / cv_max;
    
    % Normalize fitness values
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + 1e-12;
    f_norm = (popfits - f_min) / f_range;
    
    % Identify feasible solutions
    feasible_mask = cons <= 0;
    feasible_sols = popdecs(feasible_mask, :);
    
    % Calculate centers
    if ~isempty(feasible_sols)
        feasible_center = mean(feasible_sols, 1);
    else
        feasible_center = mean(popdecs, 1);
    end
    
    % Get best solution (considering constraints)
    [~, best_idx] = min(popfits + 1e6 * max(0, cons));
    x_best = popdecs(best_idx, :);
    
    % Pre-generate random indices
    rand_idx = zeros(NP, 2);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available), 2));
    end
    
    % Generate offspring
    for i = 1:NP
        % Adaptive parameters
        F1 = 0.5 * (1 + f_norm(i));
        F2 = 0.7 * (1 - f_norm(i));
        F3 = 0.3 * f_norm(i);
        CRi = 0.6 + 0.3 * (1 - cv_norm(i));
        
        % Get random vectors
        r1 = rand_idx(i,1);
        r2 = rand_idx(i,2);
        
        % Mutation
        mutant = popdecs(i,:) + ...
                 F1 * (x_best - popdecs(i,:)) + ...
                 F2 * (1 - cv_norm(i)) * (popdecs(r1,:) - popdecs(r2,:)) + ...
                 F3 * cv_norm(i) * (feasible_center - popdecs(i,:));
        
        % Crossover
        j_rand = randi(D);
        mask = rand(1,D) < CRi;
        mask(j_rand) = true;
        trial = popdecs(i,:);
        trial(mask) = mutant(mask);
        
        % Constraint repair
        if cons(i) > 0
            beta = 0.5 * cv_norm(i);
            trial = (1-beta)*trial + beta*feasible_center;
        end
        
        offspring(i,:) = trial;
    end
    
    % Boundary handling with reflection
    for j = 1:D
        below = offspring(:,j) < lb(j);
        above = offspring(:,j) > ub(j);
        offspring(below,j) = 2*lb(j) - offspring(below,j);
        offspring(above,j) = 2*ub(j) - offspring(above,j);
        
        % Additional randomization if still out of bounds
        still_below = offspring(:,j) < lb(j);
        still_above = offspring(:,j) > ub(j);
        offspring(still_below,j) = lb(j) + rand(sum(still_below),1).*(ub(j)-lb(j));
        offspring(still_above,j) = lb(j) + rand(sum(still_above),1).*(ub(j)-lb(j));
    end
end