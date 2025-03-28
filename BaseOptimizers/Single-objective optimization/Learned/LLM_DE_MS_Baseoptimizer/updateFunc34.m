function [offspring] = updateFunc34(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Find best individual based on fitness
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Calculate fitness ranks (0 to NP-1)
    [~, fit_sort_idx] = sort(popfits);
    [~, fit_ranks] = sort(fit_sort_idx);
    norm_fits = (fit_ranks - 1) / (NP - 1);  % Normalize to [0,1]
    
    % Calculate constraint violation ranks (0 to NP-1)
    [~, cons_sort_idx] = sort(abs(cons));
    [~, cons_ranks] = sort(cons_sort_idx);
    norm_cons = (cons_ranks - 1) / (NP - 1);  % Normalize to [0,1]
    
    % Generate random indices ensuring r1 ≠ r2 ≠ r3 ≠ i
    rand_idx = zeros(NP, 3);
    for i = 1:NP
        available = setdiff(1:NP, i);
        rand_idx(i,:) = available(randperm(length(available), 3));
    end
    r1 = rand_idx(:,1);
    r2 = rand_idx(:,2);
    r3 = rand_idx(:,3);
    
    % Calculate adaptive scaling factors
    F_fit = 0.7 * (1 - norm_fits);
    F_cons = 0.3 * (1 - norm_cons);
    
    % Vectorized mutation operation
    term1 = popdecs(r1, :);
    term2 = F_fit .* (x_best - popdecs(r1, :));
    term3 = F_cons .* (popdecs(r2, :) - popdecs(r3, :));
    
    offspring = term1 + term2 + term3;
end