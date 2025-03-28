function [offspring] = updateFunc3(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    epsilon = 1e-6;
    max_cons = max(cons);
    sum_cons = sum(cons);
    
    for i = 1:NP
        % Select four distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        r = candidates(randperm(NP-1, 4));
        r1 = r(1); r2 = r(2); r3 = r(3); r4 = r(4);
        
        % Fitness-weighted difference vectors
        fit_w1 = popfits(r2) / (popfits(r2) + popfits(r3) + epsilon);
        d1 = fit_w1 .* (popdecs(r2,:) - popdecs(r3,:));
        
        fit_w2 = popfits(r4) / (popfits(r4) + popfits(r1) + epsilon);
        d2 = fit_w2 .* (popdecs(r4,:) - popdecs(r1,:));
        
        % Constraint-aware scaling factors
        F1 = 0.4 + 0.6 * (cons(i) / (max_cons + 1));
        F2 = 0.8 - 0.6 * (cons(i) / (max_cons + 1));
        
        % Constraint-based perturbation
        p = (cons(i) / (sum_cons + 1)) * randn(1, D);
        
        % Generate mutant vector
        offspring(i,:) = popdecs(r1,:) + F1 .* d1 + F2 .* d2 + p;
    end
    
    % Ensure bounds are maintained (assuming [0,10] range)
    offspring = min(max(offspring, 0), 10);
end