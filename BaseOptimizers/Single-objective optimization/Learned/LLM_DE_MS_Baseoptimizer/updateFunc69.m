function [offspring] = updateFunc69(popdecs, popfits, cons)
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
    
    % Normalize fitness values
    f_min = min(popfits);
    f_max = max(popfits);
    if f_max > f_min
        w_fit = (popfits - f_min) / (f_max - f_min);
    else
        w_fit = zeros(NP, 1);
    end
    
    % Compute centroid of top 30% solutions
    [~, sorted_idx] = sort(popfits);
    top30 = sorted_idx(1:ceil(0.3*NP));
    c_top = mean(popdecs(top30, :), 1);
    
    for i = 1:NP
        % Select two distinct random indices different from i
        candidates = setdiff(1:NP, i);
        r1 = candidates(randi(length(candidates)));
        candidates = setdiff(candidates, r1);
        r2 = candidates(randi(length(candidates)));
        
        % Apply mutation
        offspring(i,:) = c_top + ...
                        F * w_cv(i) * (popdecs(r1,:) - popdecs(r2,:)) + ...
                        alpha * w_fit(i) * (x_best - popdecs(i,:));
    end
end