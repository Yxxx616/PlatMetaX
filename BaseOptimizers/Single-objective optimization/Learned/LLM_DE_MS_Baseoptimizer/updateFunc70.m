function [offspring] = updateFunc70(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Control parameters
    F = 0.8;
    alpha = 0.6;
    gamma = 0.7;
    
    % Find best solution
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Normalize constraint violations (0 to 1)
    cv_min = min(cons);
    cv_max = max(cons);
    if cv_max > cv_min
        w_cv = 1 - (cons - cv_min) / (cv_max - cv_min);
    else
        w_cv = ones(NP, 1);
    end
    
    % Normalize fitness values (0 to 1)
    f_min = min(popfits);
    f_max = max(popfits);
    if f_max > f_min
        w_fit = 1 - (popfits - f_min) / (f_max - f_min);
    else
        w_fit = ones(NP, 1);
    end
    
    % Compute centroid of top 20% solutions
    [~, sorted_idx] = sort(popfits);
    top20 = sorted_idx(1:ceil(0.2*NP));
    c_top = mean(popdecs(top20, :), 1);
    
    % Vectorized mutation
    for i = 1:NP
        % Select two distinct random indices different from i
        candidates = setdiff(1:NP, i);
        r1 = candidates(randi(length(candidates)));
        candidates = setdiff(candidates, r1);
        r2 = candidates(randi(length(candidates)));
        
        % Apply mutation
        offspring(i,:) = gamma * c_top + ...
                        F * w_cv(i) * (popdecs(r1,:) - popdecs(r2,:)) + ...
                        alpha * w_fit(i) * (x_best - popdecs(i,:));
    end
    
    % Boundary control (optional)
    % offspring = min(max(offspring, -1000), 1000);
end