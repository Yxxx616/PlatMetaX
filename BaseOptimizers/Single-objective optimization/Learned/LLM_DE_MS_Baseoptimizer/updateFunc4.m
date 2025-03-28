function [offspring] = updateFunc4(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    epsilon = 1e-6;
    max_cons = max(cons);
    sum_cons = sum(cons);
    
    for i = 1:NP
        % Select five distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        r = candidates(randperm(NP-1, 5));
        r1 = r(1); r2 = r(2); r3 = r(3); r4 = r(4); r5 = r(5);
        
        % Calculate fitness weights
        w1 = popfits(r2) / (popfits(r2) + popfits(r3) + epsilon);
        w2 = popfits(r4) / (popfits(r4) + popfits(r5) + epsilon);
        w3 = popfits(r1) / (popfits(r1) + popfits(r2) + epsilon);
        
        % Compute weighted differences
        delta1 = w1 .* (popdecs(r2,:) - popdecs(r3,:));
        delta2 = w2 .* (popdecs(r4,:) - popdecs(r5,:));
        delta3 = w3 .* (popdecs(r1,:) - popdecs(r2,:));
        
        % Constraint-aware scaling factors
        F = 0.5 * (1 + tanh(1 - cons(i)/(max_cons + 1)));
        G = 0.3 * (1 + cons(i)/(sum_cons + 1));
        
        % Generate random perturbation
        rand_perturb = randn(1, D);
        
        % Create mutant vector
        offspring(i,:) = popdecs(r1,:) + F .* delta1 + (1-F) .* delta2 + ...
                         G .* delta3 .* rand_perturb;
    end
    
    % Ensure bounds are maintained (assuming [0,10] range)
    offspring = min(max(offspring, 0), 10);
end