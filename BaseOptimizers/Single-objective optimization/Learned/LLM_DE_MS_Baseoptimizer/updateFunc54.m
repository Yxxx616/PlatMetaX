function [offspring] = updateFunc54(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Convert to maximization problem and normalize fitness
    norm_fits = -popfits;
    norm_fits = (norm_fits - min(norm_fits)) / (max(norm_fits) - min(norm_fits) + eps);
    
    % Normalize constraint violations
    norm_cons = abs(cons) / (max(abs(cons)) + eps);
    
    % Sort population by fitness
    [~, sorted_idx] = sort(popfits);
    best_idx = sorted_idx(1);
    x_best = popdecs(best_idx, :);
    
    for i = 1:NP
        % Select 3 distinct random individuals (excluding current and best)
        candidates = setdiff(1:NP, [i, best_idx]);
        idxs = candidates(randperm(length(candidates), 3));
        x1 = popdecs(idxs(1), :);
        x2 = popdecs(idxs(2), :);
        x3 = popdecs(idxs(3), :);
        
        % Get corresponding fitness values
        f = norm_fits(idxs);
        
        % Calculate weighted direction
        weights = f ./ (sum(f) + eps);
        d_fit = weights(1)*(x_best - x1) + weights(2)*(x_best - x2) + weights(3)*(x_best - x3);
        
        % Adaptive F parameter based on constraint violation
        F = 0.4 + 0.6 * exp(-5*norm_cons(i));
        
        % Select two more random vectors for perturbation
        r_idxs = candidates(randperm(length(candidates), 2));
        xr1 = popdecs(r_idxs(1), :);
        xr2 = popdecs(r_idxs(2), :);
        
        % Mutation with opposition-based perturbation
        v = x_best + F * d_fit + 0.7*F*(xr1 - xr2);
        
        % Boundary control with reflection
        lb = -100 * ones(1, D);
        ub = 100 * ones(1, D);
        v = min(max(v, lb), ub);
        
        % Reflect back if out of bounds
        mask_low = v < lb;
        mask_high = v > ub;
        v(mask_low) = 2*lb(mask_low) - v(mask_low);
        v(mask_high) = 2*ub(mask_high) - v(mask_high);
        
        offspring(i,:) = v;
    end
end