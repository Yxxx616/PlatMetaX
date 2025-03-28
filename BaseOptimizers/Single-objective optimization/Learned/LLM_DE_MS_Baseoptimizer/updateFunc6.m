function [offspring] = updateFunc6(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Parameters
    F = 0.5;
    lambda = 0.1;
    top_ratio = 0.3;
    
    % Normalize constraint violations
    c_max = max(abs(cons));
    c_norm = abs(cons) ./ max(c_max, 1e-12);
    
    % Sort by fitness
    [~, sorted_idx] = sort(popfits);
    top_count = round(NP * top_ratio);
    
    % Find best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    for i = 1:NP
        % Select r1 from top individuals
        r1 = sorted_idx(randi(top_count));
        while r1 == i
            r1 = sorted_idx(randi(top_count));
        end
        
        % Select r2 considering both fitness and constraints
        weights = 0.7 * (1:NP)'/NP + 0.3 * (1 - c_norm);
        weights = weights / sum(weights);
        r2 = randsample(NP, 1, true, weights);
        while r2 == i || r2 == r1
            r2 = randsample(NP, 1, true, weights);
        end
        
        % Mutation with constraint adaptation
        offspring(i,:) = x_best + F * (popdecs(r1,:) - popdecs(r2,:)) + ...
                        lambda * (1 - c_norm(i)) * randn(1, D);
    end
end