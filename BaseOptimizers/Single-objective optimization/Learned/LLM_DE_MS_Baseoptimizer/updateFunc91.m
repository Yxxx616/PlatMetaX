function [offspring] = updateFunc91(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Find best constraint violation
    [c_best, ~] = min(cons);
    
    for i = 1:NP
        % Select three distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        idx = candidates(randperm(NP-1, 3));
        r1 = idx(1); r2 = idx(2); r3 = idx(3);
        
        % Fitness-weighted difference vector
        f_sum = popfits(r1) + popfits(r2);
        if f_sum > 0
            d = (popfits(r2)/f_sum) * (popdecs(r1,:) - popdecs(r2,:));
        else
            d = popdecs(r1,:) - popdecs(r2,:);
        end
        
        % Constraint-aware scaling factor
        if cons(r3) > 0
            F = 0.5 * (1 + c_best / cons(r3));
        else
            F = 0.8; % Default when no constraint violation
        end
        
        % Generate mutant vector
        v = popdecs(r3,:) + F * d;
        
        % Apply bound constraints
        v = max(min(v, 10), -10);
        
        offspring(i,:) = v;
    end
end