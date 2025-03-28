function [offspring] = updateFunc55(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness (convert to maximization)
    norm_fits = -popfits;
    norm_fits = (norm_fits - min(norm_fits)) / (max(norm_fits) - min(norm_fits) + eps);
    
    % Calculate weighted centroid
    weights = norm_fits ./ (sum(norm_fits) + eps);
    centroid = weights' * popdecs;
    
    % Find best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Normalize constraint violations
    norm_cons = abs(cons) / (max(abs(cons)) + eps);
    mean_cons = mean(norm_cons);
    std_cons = std(norm_cons) + eps;
    
    for i = 1:NP
        % Adaptive F parameter based on constraints
        F = 0.5 * (1 + tanh((norm_cons(i) - mean_cons)/std_cons));
        
        % Select two distinct random individuals (excluding current)
        candidates = setdiff(1:NP, i);
        idxs = candidates(randperm(length(candidates), 2));
        x1 = popdecs(idxs(1), :);
        x2 = popdecs(idxs(2), :);
        
        % Mutation with directional components
        v = x_best + F*(x1 - x2) + (1-F)*(centroid - popdecs(i,:));
        
        % Boundary handling
        out_of_bounds = (v < lb) | (v > ub);
        if any(out_of_bounds)
            if rand < 0.5
                % Random reinitialization for out-of-bounds dimensions
                v(out_of_bounds) = lb(out_of_bounds) + rand(1, sum(out_of_bounds)) .* ...
                                  (ub(out_of_bounds) - lb(out_of_bounds));
            else
                % Reflection
                low_mask = v < lb;
                v(low_mask) = 2*lb(low_mask) - v(low_mask);
                high_mask = v > ub;
                v(high_mask) = 2*ub(high_mask) - v(high_mask);
            end
        end
        
        offspring(i,:) = v;
    end
end