function [offspring] = updateFunc23(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Find best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Normalize constraints and fitness
    max_cons = max(abs(cons));
    min_fit = min(popfits);
    max_fit = max(popfits);
    range_fit = max(max_fit - min_fit, 1e-12); % Avoid division by zero
    
    % Pre-compute all random indices (vectorized)
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    r3 = randi(NP, NP, 1);
    r4 = randi(NP, NP, 1);
    
    % Ensure all indices are distinct
    for i = 1:NP
        while r1(i) == r2(i) || r1(i) == r3(i) || r1(i) == r4(i) || ...
              r2(i) == r3(i) || r2(i) == r4(i) || r3(i) == r4(i)
            r1(i) = randi(NP);
            r2(i) = randi(NP);
            r3(i) = randi(NP);
            r4(i) = randi(NP);
        end
    end
    
    % Compute scaling factors
    F = 0.5 + 0.3 * (1 - abs(cons) ./ max_cons);
    alpha = 0.5 * (popfits - min_fit) ./ range_fit;
    
    % Vectorized mutation
    for i = 1:NP
        offspring(i,:) = x_best + ...
                        F(i) * (popdecs(r1(i),:) - popdecs(r2(i),:)) + ...
                        alpha(i) * (popdecs(r3(i),:) - popdecs(r4(i),:));
    end
    
    % Boundary control with reflection
    lb = -600 * ones(1, D);
    ub = 600 * ones(1, D);
    
    for i = 1:NP
        for j = 1:D
            if offspring(i,j) < lb(j)
                offspring(i,j) = lb(j) + rand() * (popdecs(i,j) - lb(j));
            elseif offspring(i,j) > ub(j)
                offspring(i,j) = ub(j) - rand() * (ub(j) - popdecs(i,j));
            end
        end
    end
end