function [offspring] = updateFunc32(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Find best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Normalize fitness values
    f_min = min(popfits);
    f_max = max(popfits);
    if f_max == f_min
        norm_fits = zeros(NP, 1);
    else
        norm_fits = (popfits - f_min) / (f_max - f_min);
    end
    
    % Normalize constraint violations
    max_cons = max(abs(cons));
    if max_cons == 0
        norm_cons = zeros(NP, 1);
    else
        norm_cons = abs(cons) / max_cons;
    end
    
    % Generate random indices matrix
    [~, rand_idx] = sort(rand(NP, NP), 2);
    mask = rand_idx == (1:NP)';
    rand_idx(mask) = rand_idx(mask) + 1;
    rand_idx = mod(rand_idx-1, NP) + 1;
    
    % Get r1, r2, r3 indices
    r1 = rand_idx(:, 1);
    r2 = rand_idx(:, 2);
    r3 = rand_idx(:, 3);
    
    % Calculate scaling factors
    F1 = 0.5 * (1 - norm_fits);
    F2 = 0.5 * (1 - norm_cons);
    
    % Vectorized mutation
    term1 = popdecs(r1, :);
    term2 = F1 .* (x_best - popdecs(r1, :));
    term3 = F2 .* (popdecs(r2, :) - popdecs(r3, :));
    
    offspring = term1 + term2 + term3;
end