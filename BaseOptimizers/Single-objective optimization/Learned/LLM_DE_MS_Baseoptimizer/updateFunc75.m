function [offspring] = updateFunc75(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Find best solutions
    [~, best_fit_idx] = min(popfits);
    [~, best_cons_idx] = min(cons);
    x_best_fit = popdecs(best_fit_idx, :);
    x_best_cons = popdecs(best_cons_idx, :);
    
    % Calculate population statistics
    pop_mean = mean(popdecs, 1);
    pop_std = std(popdecs, 0, 1);
    max_std = max(pop_std);
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons));
    
    % Adaptive parameters
    F_base = 0.5 + 0.3 * (popfits - f_min) ./ (f_max - f_min + eps);
    beta = 0.2 * (1 - abs(cons) ./ (c_max + eps));
    gamma = 0.1 * pop_std ./ (max_std + eps);
    
    for i = 1:NP
        % Select 3 distinct random vectors
        candidates = setdiff(1:NP, [i, best_fit_idx, best_cons_idx]);
        r = candidates(randperm(length(candidates), 3));
        x_r1 = popdecs(r(1), :);
        x_r2 = popdecs(r(2), :);
        x_r3 = popdecs(r(3), :);
        
        % Calculate mutation components
        fit_guide = F_base(i) * (x_best_fit - popdecs(i,:));
        diff_term = 0.8 * rand() * (x_r1 - x_r2);
        cons_guide = beta(i) * (x_best_cons - popdecs(i,:));
        diversity_term = gamma .* (x_r3 - pop_mean);
        
        % Combine components
        offspring(i,:) = popdecs(i,:) + fit_guide + diff_term + cons_guide + diversity_term;
    end
    
    % Boundary handling with adaptive reflection
    lb = -1000;
    ub = 1000;
    for d = 1:D
        % Calculate reflection factor based on dimension diversity
        reflect_factor = 0.4 + 0.3 * (pop_std(d) / (pop_std(d) + 100));
        
        % Handle out-of-bounds
        below = offspring(:,d) < lb;
        above = offspring(:,d) > ub;
        
        % Adaptive reflection
        offspring(below,d) = lb + reflect_factor * (lb - offspring(below,d));
        offspring(above,d) = ub - reflect_factor * (offspring(above,d) - ub);
        
        % Final clipping
        offspring(:,d) = min(max(offspring(:,d), lb), ub);
    end
end