function [offspring] = updateFunc61(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Calculate feasibility (constraint violation <= 0)
    feasible = all(cons <= 0, 2);
    
    % Sort population by fitness (minimization)
    [~, sorted_idx] = sort(popfits);
    pop_sorted = popdecs(sorted_idx, :);
    fits_sorted = popfits(sorted_idx);
    cons_sorted = cons(sorted_idx, :);
    
    % Get best solution (considering feasibility)
    if any(feasible)
        feasible_fits = popfits(feasible);
        [~, best_idx] = min(feasible_fits);
        x_best = popdecs(feasible(best_idx), :);
    else
        [~, best_idx] = min(popfits);
        x_best = popdecs(best_idx, :);
    end
    
    % Population statistics
    x_avg = mean(popdecs, 1);
    f_avg = mean(popfits);
    f_max = max(popfits);
    f_min = min(popfits);
    
    % Segment population into three parts
    n1 = floor(0.3*NP);
    n2 = floor(0.7*NP);
    
    for i = 1:NP
        % Select donors from different segments
        r1 = randi([1, n1]);
        r2 = randi([n1+1, n2]);
        r3 = randi([n2+1, NP]);
        
        % Adaptive scaling factor
        F = 0.5 * (1 + (f_avg - popfits(i)) / (f_max - f_min + eps));
        
        % Mutation
        offspring(i,:) = x_best + F*(pop_sorted(r1,:) - pop_sorted(r2,:)) + ...
                         (1-F)*(pop_sorted(r3,:) - x_avg);
        
        % Boundary control
        offspring(i,:) = min(max(offspring(i,:), -500), 500);
    end
end