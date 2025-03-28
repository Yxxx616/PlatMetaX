function [offspring] = updateFunc79(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Parameters
    F = 0.7;
    alpha = 0.4;
    beta = 0.5;
    
    % Transform fitness (assuming minimization)
    transformed_fits = -popfits;
    min_fit = min(transformed_fits);
    max_fit = max(transformed_fits);
    norm_fits = (transformed_fits - min_fit) / (max_fit - min_fit + eps);
    
    % Transform constraints
    abs_cons = abs(cons);
    min_cons = min(abs_cons);
    max_cons = max(abs_cons);
    norm_cons = (abs_cons - min_cons) / (max_cons - min_cons + eps);
    
    for i = 1:NP
        % Select six distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        r = candidates(randperm(length(candidates), 6));
        r1 = r(1); r2 = r(2); r3 = r(3); r4 = r(4); r5 = r(5); r6 = r(6);
        
        % Base vector (fitness-weighted)
        w1 = norm_fits(r2) / (norm_fits(r1) + norm_fits(r2) + eps);
        w2 = 1 - w1;
        base_vec = w1 * popdecs(r1,:) + w2 * popdecs(r2,:);
        
        % Differential component (fitness-directed)
        diff_vec = F * (norm_fits(r3) * popdecs(r3,:) - norm_fits(r4) * popdecs(r4,:));
        
        % Constraint component (magnitude-sensitive)
        cons_sign = sign(cons(r5));
        cons_mag = norm_cons(r5)^beta;
        cons_vec = alpha * cons_sign * cons_mag * (popdecs(r5,:) - popdecs(r6,:));
        
        % Combine components
        offspring(i,:) = base_vec + diff_vec + cons_vec;
    end
    
    % Boundary control
    lb = -100 * ones(1,D);
    ub = 100 * ones(1,D);
    offspring = min(max(offspring, lb), ub);
end