function [offspring] = updateFunc31(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    F = 0.5;
    
    % Find best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Normalize constraint violations
    max_cons = max(abs(cons));
    if max_cons == 0
        max_cons = 1; % avoid division by zero
    end
    norm_cons = abs(cons) / max_cons;
    
    for i = 1:NP
        % Select three distinct random indices
        candidates = 1:NP;
        candidates(i) = [];
        r = candidates(randperm(length(candidates), 3));
        r1 = r(1); r2 = r(2); r3 = r(3);
        
        % Constraint-aware mutation
        scale_factor = F * (1 - norm_cons(i));
        offspring(i,:) = popdecs(r1,:) + F*(x_best - popdecs(r1,:)) + ...
                         scale_factor*(popdecs(r2,:) - popdecs(r3,:));
    end
end