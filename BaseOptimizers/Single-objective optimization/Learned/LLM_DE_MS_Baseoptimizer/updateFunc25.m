function [offspring] = updateFunc25(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Find best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Normalize constraints and fitness
    max_cons = max(abs(cons));
    min_fit = min(popfits);
    max_fit = max(popfits);
    range_fit = max(max_fit - min_fit, 1e-12);
    
    beta = 1 - abs(cons) ./ max(max_cons, 1e-12); % Constraint factor
    alpha = (popfits - min_fit) ./ range_fit;     % Fitness factor
    
    % Pre-compute all random indices (vectorized)
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    r3 = randi(NP, NP, 1);
    r4 = randi(NP, NP, 1);
    
    % Ensure all indices are distinct and not current index
    for i = 1:NP
        while r1(i) == i || r2(i) == i || r3(i) == i || r4(i) == i || ...
              r1(i) == r2(i) || r1(i) == r3(i) || r1(i) == r4(i) || ...
              r2(i) == r3(i) || r2(i) == r4(i) || r3(i) == r4(i)
            r1(i) = randi(NP);
            r2(i) = randi(NP);
            r3(i) = randi(NP);
            r4(i) = randi(NP);
        end
    end
    
    % Compute adaptive scaling factors
    F1 = 0.5 * (1 + beta);
    F2 = 0.3 * (1 - alpha);
    F3 = 0.2 * beta .* alpha;
    
    % Vectorized mutation
    for i = 1:NP
        offspring(i,:) = popdecs(i,:) + ...
                        F1(i) * (x_best - popdecs(i,:)) + ...
                        F2(i) * (popdecs(r1(i),:) - popdecs(r2(i),:)) + ...
                        F3(i) * (popdecs(r3(i),:) - popdecs(r4(i),:));
    end
    
    % Boundary control with directed repair
    lb = -600 * ones(1, D);
    ub = 600 * ones(1, D);
    
    for i = 1:NP
        for j = 1:D
            if offspring(i,j) < lb(j) || offspring(i,j) > ub(j)
                % Repair towards best solution with random scaling
                offspring(i,j) = popdecs(i,j) + rand() * (x_best(j) - popdecs(i,j));
            end
        end
    end
end