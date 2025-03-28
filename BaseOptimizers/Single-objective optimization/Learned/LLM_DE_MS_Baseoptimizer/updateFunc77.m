function [offspring] = updateFunc77(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Parameters
    F = 0.8;
    alpha = 0.5;
    
    % Normalize fitness and constraints
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    
    for i = 1:NP
        % Select three distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        r = candidates(randperm(length(candidates), 3));
        r1 = r(1); r2 = r(2); r3 = r(3);
        
        % Fitness-weighted difference component
        fit_weight = norm_fits(r2) / (norm_fits(r2) + norm_fits(r3) + eps);
        diff_vec = popdecs(r2,:) - popdecs(r3,:);
        
        % Constraint-guided perturbation
        cons_weight = norm_cons(r1) / (norm_cons(r1) + norm_cons(r2) + eps);
        pert_vec = popdecs(r1,:) - popdecs(r2,:);
        
        % Combine components
        offspring(i,:) = popdecs(r1,:) + F * fit_weight * diff_vec + alpha * cons_weight * pert_vec;
    end
    
    % Boundary control
    lb = -100 * ones(1,D);
    ub = 100 * ones(1,D);
    offspring = min(max(offspring, lb), ub);
end