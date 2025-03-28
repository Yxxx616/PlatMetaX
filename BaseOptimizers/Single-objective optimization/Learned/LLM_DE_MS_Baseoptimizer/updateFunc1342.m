% MATLAB Code
function [offspring] = updateFunc1342(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Calculate constraint violation normalization
    cv_abs = abs(cons);
    cv_max = max(cv_abs) + 1e-12;
    cv_norm = cv_abs / cv_max;
    
    % Calculate fitness normalization
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + 1e-12;
    f_norm = (popfits - f_min) / f_range;
    
    % Identify top 30% solutions
    [~, sorted_idx] = sortrows([popfits, cv_abs], [1 2]);
    elite_num = max(2, ceil(0.3*NP));
    elite = popdecs(sorted_idx(1:elite_num), :);
    x_best = popdecs(sorted_idx(1), :);
    center = mean(elite, 1);
    
    % Pre-generate random indices
    rand_idx = zeros(NP, 2);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available), 2));
    end
    
    % Generate offspring
    for i = 1:NP
        % Calculate adaptive parameters
        F1 = 0.4 * (1 + f_norm(i));
        F2 = 0.6 * (1 - cv_norm(i));
        F3 = 0.2 * cv_norm(i);
        CRi = 0.5 + 0.4 * (1 - cv_norm(i));
        
        % Get random vectors
        r1 = rand_idx(i,1);
        r2 = rand_idx(i,2);
        
        % Mutation
        mutant = popdecs(i,:) + ...
                 F1 * (x_best - popdecs(i,:)) + ...
                 F2 * (popdecs(r1,:) - popdecs(r2,:)) + ...
                 F3 * (center - popdecs(i,:));
        
        % Crossover
        j_rand = randi(D);
        mask = rand(1,D) < CRi;
        mask(j_rand) = true;
        trial = popdecs(i,:);
        trial(mask) = mutant(mask);
        
        % Constraint repair if needed
        if cons(i) > 0
            beta = 0.5 * cv_norm(i);
            trial = (1-beta)*trial + beta*center;
        end
        
        offspring(i,:) = trial;
    end
    
    % Boundary handling
    for j = 1:D
        below = offspring(:,j) < lb(j);
        above = offspring(:,j) > ub(j);
        offspring(below,j) = lb(j) + rand(sum(below),1) .* (ub(j)-lb(j));
        offspring(above,j) = lb(j) + rand(sum(above),1) .* (ub(j)-lb(j));
    end
end