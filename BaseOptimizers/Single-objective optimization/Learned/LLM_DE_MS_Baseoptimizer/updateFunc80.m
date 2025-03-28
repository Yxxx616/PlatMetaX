function [offspring] = updateFunc80(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Parameters
    F = 0.8;  % Scaling factor
    alpha = 0.5;  % Constraint influence
    beta = 0.6;  % Non-linearity for constraints
    
    % Normalize fitness (assuming minimization)
    transformed_fits = -popfits;
    min_fit = min(transformed_fits);
    max_fit = max(transformed_fits);
    range_fit = max_fit - min_fit + eps;
    
    % Precompute constraint terms
    cons_sign = sign(cons);
    abs_cons = abs(cons);
    
    for i = 1:NP
        % Select six distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        r = candidates(randperm(length(candidates), 6));
        r1 = r(1); r2 = r(2); r3 = r(3); r4 = r(4); r5 = r(5); r6 = r(6);
        
        % Base vector (fitness-weighted)
        w1 = transformed_fits(r2) / (transformed_fits(r1) + transformed_fits(r2) + eps);
        base_vec = w1 * popdecs(r1,:) + (1 - w1) * popdecs(r2,:);
        
        % Differential vector (adaptive scaling)
        fit_diff = (transformed_fits(r3) - transformed_fits(r4)) / range_fit;
        diff_vec = F * fit_diff * (popdecs(r3,:) - popdecs(r4,:));
        
        % Constraint vector (non-linear transformation)
        cons_term = alpha * tanh(abs_cons(r5)) .* (abs_cons(r5).^beta);
        cons_vec = cons_sign(r5) .* cons_term .* (popdecs(r5,:) - popdecs(r6,:));
        
        % Combine components
        offspring(i,:) = base_vec + diff_vec + cons_vec;
    end
    
    % Boundary control
    lb = -100 * ones(1,D);
    ub = 100 * ones(1,D);
    offspring = min(max(offspring, lb), ub);
end