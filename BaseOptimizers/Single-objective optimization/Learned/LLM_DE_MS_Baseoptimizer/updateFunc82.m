function [offspring] = updateFunc82(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Calculate centroid
    centroid = mean(popdecs, 1);
    
    % Calculate distance to centroid
    dists = sqrt(sum((popdecs - centroid).^2, 2));
    norm_dists = dists / max(dists);
    
    % Calculate selection weights combining fitness and constraints
    weights = 1./(1 + abs(popfits) + abs(cons) + norm_dists);
    weights = weights / sum(weights);
    
    % Find best individual
    [~, best_idx] = max(weights);
    
    % Base scaling parameters
    F_base = 0.5;
    eta = 0.2;
    
    for i = 1:NP
        % Tournament selection for base vector
        candidates = setdiff(1:NP, i);
        [~, idx] = max(weights(candidates(randperm(length(candidates), 3))));
        base_idx = candidates(idx);
        
        % Select difference vectors
        candidates = setdiff(1:NP, [i, base_idx]);
        r = randsample(candidates, 4, true, weights(candidates));
        r1 = r(1); r2 = r(2); r3 = r(3); r4 = r(4);
        
        % Calculate weighted differences
        diff1 = popdecs(r1,:) - popdecs(r2,:);
        diff2 = popdecs(r3,:) - popdecs(r4,:);
        
        % Adaptive scaling factor
        F_i = F_base + 0.5 * (weights(i) / max(weights));
        
        % Generate mutation vector
        mutant = popdecs(base_idx,:) + F_i * (diff1 + diff2) + ...
                 eta * (popdecs(best_idx,:) - popdecs(i,:));
        
        offspring(i,:) = mutant;
    end
end