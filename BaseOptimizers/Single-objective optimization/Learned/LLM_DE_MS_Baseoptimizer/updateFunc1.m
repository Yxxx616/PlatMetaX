function [offspring] = updateFunc1(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    epsilon = 1e-6;
    
    for i = 1:NP
        % Select three distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        r = candidates(randperm(NP-1, 3));
        r1 = r(1); r2 = r(2); r3 = r(3);
        
        % Fitness-weighted difference vector
        d = (popfits(r2) - popfits(r3)) .* (popdecs(r2,:) - popdecs(r3,:));
        
        % Constraint-aware scaling factor
        F = 0.5 * (1 + cons(r1) / (cons(r2) + cons(r3) + epsilon));
        
        % Random perturbation based on constraint violation
        rand_perturb = 0.5 * cons(i) .* rand(1, D);
        
        % Generate mutant vector
        offspring(i,:) = popdecs(r1,:) + F .* d + rand_perturb;
    end
    
    % Ensure bounds are maintained (assuming [0,10] range)
    offspring = min(max(offspring, 0), 10);
end