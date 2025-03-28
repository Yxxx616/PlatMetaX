% MATLAB Code
function [offspring] = updateFunc1345(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    eps = 1e-12;
    
    % Normalize constraints
    cv_abs = abs(cons);
    cv_max = max(cv_abs) + eps;
    cv_norm = cv_abs / cv_max;
    
    % Normalize fitness
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    f_norm = (popfits - f_min) / f_range;
    
    % Identify feasible solutions and center
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        feasible_center = mean(popdecs(feasible_mask, :), 1);
    else
        feasible_center = mean(popdecs, 1);
    end
    
    % Find best solution considering constraints
    [~, best_idx] = min(popfits + 1e6 * max(0, cons));
    x_best = popdecs(best_idx, :);
    
    % Pre-generate random indices
    rand_idx1 = zeros(NP, 1);
    rand_idx2 = zeros(NP, 1);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx1(i) = available(randi(length(available)));
        rand_idx2(i) = available(randi(length(available)));
    end
    
    % Generate offspring
    for i = 1:NP
        % Mutation factors
        F1 = 0.8 * (1 - f_norm(i));
        F2 = 0.6 * f_norm(i);
        F3 = 0.4 * cv_norm(i);
        
        % Mutation
        mutant = popdecs(i,:) + ...
                 F1 * (x_best - popdecs(i,:)) + ...
                 F2 * (popdecs(rand_idx1(i),:) - popdecs(rand_idx2(i),:)) + ...
                 F3 * (feasible_center - popdecs(i,:));
        
        % Crossover
        CR = 0.9 - 0.5 * cv_norm(i);
        j_rand = randi(D);
        mask = rand(1,D) < CR;
        mask(j_rand) = true;
        trial = popdecs(i,:);
        trial(mask) = mutant(mask);
        
        % Constraint repair
        if cons(i) > 0
            beta = 0.3 * cv_norm(i);
            trial = (1-beta)*trial + beta*feasible_center;
        end
        
        offspring(i,:) = trial;
    end
    
    % Boundary handling with reflection
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