function [offspring] = updateFunc81(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Calculate selection weights
    weights = 1./(1 + abs(popfits) + abs(cons));
    weights = weights / sum(weights); % Normalize
    
    % Find best individual index
    [~, best_idx] = max(weights);
    
    F = 0.8;
    lambda = 0.5;
    
    for i = 1:NP
        % Select three distinct parents using weights
        candidates = setdiff(1:NP, i);
        r = randsample(candidates, 3, true, weights(candidates));
        r1 = r(1); r2 = r(2); r3 = r(3);
        
        % Generate mutation vector
        mutant = popdecs(r1,:) + F*(popdecs(r2,:) - popdecs(r3,:)) + ...
                 lambda*(popdecs(best_idx,:) - popdecs(i,:));
        
        offspring(i,:) = mutant;
    end
end