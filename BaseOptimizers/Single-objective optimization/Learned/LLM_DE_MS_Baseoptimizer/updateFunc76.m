function [offspring] = updateFunc76(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Parameters
    F = 0.8;
    lambda = 0.2;
    
    % Find best individual (considering constraints)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits .* feasible_mask);
    else
        [~, best_idx] = min(cons);
    end
    x_best = popdecs(best_idx, :);
    
    for i = 1:NP
        % Select three distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        r = candidates(randperm(length(candidates), 3));
        r1 = r(1); r2 = r(2); r3 = r(3);
        
        % Fitness-weighted difference vector
        delta = (popfits(r2) - popfits(r3)) * (popdecs(r2,:) - popdecs(r3,:));
        
        % Mutation vector
        offspring(i,:) = popdecs(r1,:) + F * delta + lambda * (x_best - popdecs(r1,:));
    end
    
    % Boundary control
    lb = -100 * ones(1,D);
    ub = 100 * ones(1,D);
    offspring = min(max(offspring, lb), ub);
end