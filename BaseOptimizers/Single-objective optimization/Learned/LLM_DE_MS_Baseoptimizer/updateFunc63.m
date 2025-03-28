function [offspring] = updateFunc63(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Calculate feasibility and population statistics
    feasible = all(cons <= 0, 2);
    x_avg = mean(popdecs, 1);
    f_avg = mean(popfits);
    f_max = max(popfits);
    f_min = min(popfits);
    phi_avg = mean(cons);
    
    % Sort population by fitness
    [~, sorted_idx] = sort(popfits);
    pop_sorted = popdecs(sorted_idx, :);
    fits_sorted = popfits(sorted_idx);
    cons_sorted = cons(sorted_idx);
    
    % Get best solution (considering feasibility)
    if any(feasible)
        feasible_idx = find(feasible);
        [~, best_local_idx] = min(popfits(feasible));
        x_best = popdecs(feasible_idx(best_local_idx), :);
    else
        [~, best_idx] = min(popfits);
        x_best = popdecs(best_idx, :);
    end
    
    % Get worst solution
    [~, worst_idx] = max(popfits);
    x_worst = popdecs(worst_idx, :);
    
    % Segment population indices
    n1 = floor(0.3*NP);
    n2 = floor(0.7*NP);
    
    for i = 1:NP
        % Select donors from different segments
        r1 = randi([1, n1]);
        r2 = randi([n1+1, n2]);
        r3 = randi([n2+1, NP]);
        
        % Adaptive scaling factors
        F1 = 0.5 * (1 + (f_avg - popfits(i)) / (f_max - f_min + eps));
        F2 = 0.8 * tanh(cons(i) / (phi_avg + eps));
        
        % Constraint-based mutation
        if cons(i) <= phi_avg
            offspring(i,:) = x_best + F1*(pop_sorted(r1,:) - pop_sorted(r2,:)) + ...
                            F2*(pop_sorted(r3,:) - x_avg);
        else
            offspring(i,:) = x_avg + F1*(pop_sorted(r1,:) - pop_sorted(r2,:)) + ...
                            F2*(x_best - x_worst);
        end
        
        % Enhanced boundary control
        for j = 1:D
            if offspring(i,j) < -500
                lb = min(-500, x_best(j));
                offspring(i,j) = lb + rand * abs(x_best(j) - (-500));
            elseif offspring(i,j) > 500
                ub = max(500, x_best(j));
                offspring(i,j) = ub - rand * abs(500 - x_best(j));
            end
        end
    end
end