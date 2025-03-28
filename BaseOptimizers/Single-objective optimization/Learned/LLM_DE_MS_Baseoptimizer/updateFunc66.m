function [offspring] = updateFunc66(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Parameters
    F = 0.8;
    best_idx = find(popfits == min(popfits), 1);
    x_best = popdecs(best_idx, :);
    
    % Normalize constraint violations
    cv_norm = cons - min(cons);
    if max(cv_norm) > 0
        cv_norm = cv_norm / max(cv_norm);
    end
    
    % Sort by fitness for selection
    [~, sorted_idx] = sort(popfits);
    top30 = sorted_idx(1:ceil(0.3*NP));
    
    for i = 1:NP
        % Select r1 as best in neighborhood (5 closest by Euclidean distance)
        dists = sqrt(sum((popdecs - popdecs(i,:)).^2, 2));
        [~, dist_idx] = sort(dists);
        neighborhood = dist_idx(2:6); % exclude self
        [~, best_in_nbhd] = min(popfits(neighborhood));
        r1 = neighborhood(best_in_nbhd);
        
        % Select r2 from top 30% solutions
        available = setdiff(top30, [i, r1]);
        if isempty(available)
            r2 = randi(NP);
        else
            r2 = available(randi(length(available)));
        end
        
        % Select r3 from solutions with low constraint violation
        low_cv = find(cv_norm < 0.3);
        low_cv = setdiff(low_cv, [i, r1, r2]);
        if isempty(low_cv)
            r3 = randi(NP);
        else
            r3 = low_cv(randi(length(low_cv)));
        end
        
        % Adaptive weight based on constraint violation
        lambda = 0.5 * (1 + cv_norm(i));
        
        % Mutation
        offspring(i,:) = popdecs(r1,:) + F * (popdecs(r2,:) - popdecs(r3,:)) + ...
                         lambda * (x_best - popdecs(i,:));
    end
end