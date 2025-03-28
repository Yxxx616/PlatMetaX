function [offspring] = updateFunc78(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Parameters
    F = 0.8;
    alpha = 0.5;
    
    % Normalize fitness (inverted since lower is better for minimization)
    norm_fits = -popfits;
    norm_fits = (norm_fits - min(norm_fits)) / (max(norm_fits) - min(norm_fits) + eps);
    
    % Normalize constraints (absolute value since violation magnitude matters)
    norm_cons = abs(cons);
    norm_cons = (norm_cons - min(norm_cons)) / (max(norm_cons) - min(norm_cons) + eps);
    
    for i = 1:NP
        % Select six distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        r = candidates(randperm(length(candidates), 6));
        r1 = r(1); r2 = r(2); r3 = r(3); r4 = r(4); r5 = r(5); r6 = r(6);
        
        % Base vector weights
        w1 = norm_fits(r1) / (norm_fits(r1) + norm_fits(r2) + eps);
        w2 = 1 - w1;
        base_vec = w1 * popdecs(r1,:) + w2 * popdecs(r2,:);
        
        % Differential component
        fit_weight = norm_fits(r3) / (norm_fits(r3) + norm_fits(r4) + eps);
        diff_vec = fit_weight * (popdecs(r3,:) - popdecs(r4,:));
        
        % Constraint component
        cons_weight = norm_cons(r5) / (norm_cons(r5) + norm_cons(r6) + eps);
        cons_vec = cons_weight * (popdecs(r5,:) - popdecs(r6,:));
        
        % Combine components
        offspring(i,:) = base_vec + F * diff_vec + alpha * cons_vec;
    end
    
    % Boundary control
    lb = -100 * ones(1,D);
    ub = 100 * ones(1,D);
    offspring = min(max(offspring, lb), ub);
end