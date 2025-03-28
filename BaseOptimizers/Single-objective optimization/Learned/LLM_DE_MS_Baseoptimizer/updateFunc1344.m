% MATLAB Code
function [offspring] = updateFunc1344(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Normalize constraint violations (0 to 1)
    cv_abs = abs(cons);
    cv_max = max(cv_abs) + 1e-12;
    cv_norm = cv_abs / cv_max;
    
    % Normalize fitness values (0 to 1)
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
    
    % Pre-generate random indices (vectorized)
    [~, sorted_idx] = sort(popfits);
    rand_idx1 = zeros(NP, 1);
    rand_idx2 = zeros(NP, 1);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx1(i) = available(randi(length(available)));
        rand_idx2(i) = sorted_idx(randi(ceil(NP/2))); % Prefer better solutions
    end
    
    % Generate offspring (vectorized)
    F1 = 0.5 * (1 + f_norm);
    F2 = 0.7 * (1 - f_norm);
    F3 = 0.3 * f_norm;
    CR = 0.6 + 0.3 * (1 - cv_norm);
    
    for i = 1:NP
        % Mutation
        mutant = popdecs(i,:) + ...
                 F1(i) * (x_best - popdecs(i,:)) + ...
                 F2(i) * (1 - cv_norm(i)) * (popdecs(rand_idx1(i),:) - popdecs(rand_idx2(i),:)) + ...
                 F3(i) * cv_norm(i) * (feasible_center - popdecs(i,:));
        
        % Crossover
        j_rand = randi(D);
        mask = rand(1,D) < CR(i);
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
    
    % Boundary handling with reflection and randomization
    for j = 1:D
        % Reflection
        below = offspring(:,j) < lb(j);
        above = offspring(:,j) > ub(j);
        offspring(below,j) = 2*lb(j) - offspring(below,j);
        offspring(above,j) = 2*ub(j) - offspring(above,j);
        
        % Randomization if still out of bounds
        still_out = (offspring(:,j) < lb(j)) | (offspring(:,j) > ub(j));
        offspring(still_out,j) = lb(j) + rand(sum(still_out),1).*(ub(j)-lb(j));
    end
end