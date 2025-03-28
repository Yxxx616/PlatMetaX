function [offspring] = updateFunc68(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Parameters
    F = 0.8;
    alpha = 0.5;
    
    % Find best solution
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Normalize constraint violations
    cv_min = min(cons);
    cv_max = max(cons);
    if cv_max > cv_min
        w_cv = 1 - (cons - cv_min) / (cv_max - cv_min);
    else
        w_cv = ones(NP, 1);
    end
    
    % Compute pairwise distances
    dists = zeros(NP);
    for i = 1:NP
        dists(i,:) = sqrt(sum((popdecs - popdecs(i,:)).^2, 2));
    end
    dists(logical(eye(NP))) = inf; % exclude self
    min_dists = min(dists, [], 2);
    d_norm = min_dists / max(min_dists); % normalize distances
    
    % Identify top 30% solutions
    [~, sorted_idx] = sort(popfits);
    top30 = sorted_idx(1:ceil(0.3*NP));
    
    for i = 1:NP
        % Select r1 from top 30%
        r1 = top30(randi(length(top30)));
        
        % Select distinct r2, r3
        candidates = setdiff(1:NP, [i, r1]);
        r2 = candidates(randi(length(candidates)));
        candidates = setdiff(candidates, r2);
        r3 = candidates(randi(length(candidates)));
        
        % Apply mutation with adaptive weights
        offspring(i,:) = popdecs(r1,:) + ...
                        F * (popdecs(r2,:) - popdecs(r3,:)) .* w_cv(i) + ...
                        alpha * (x_best - popdecs(i,:)) .* d_norm(i);
    end
end