function [offspring] = updateFunc74(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Adaptive parameters
    F1 = 0.6 + 0.2 * rand(NP, 1);  % F between 0.6-0.8
    F2 = 0.4 + 0.2 * rand(NP, 1);  % F between 0.4-0.6
    beta = 0.05 + 0.1 * rand(NP, 1); % Constraint influence factor
    
    % Find best solutions
    [~, best_fit_idx] = min(popfits);
    [~, best_cons_idx] = min(cons);
    x_best_fit = popdecs(best_fit_idx, :);
    x_best_cons = popdecs(best_cons_idx, :);
    
    % Calculate population diversity
    pop_std = std(popdecs, 0, 1);
    diversity = mean(pop_std);
    alpha = 0.3 + 0.5 * (diversity / (diversity + 100)); % Adaptive weight
    
    % Combined best solution
    x_best = alpha * x_best_fit + (1 - alpha) * x_best_cons;
    c_min = min(cons);
    
    for i = 1:NP
        % Select 5 distinct random vectors
        candidates = setdiff(1:NP, [i, best_fit_idx, best_cons_idx]);
        r = candidates(randperm(length(candidates), 5));
        x_r1 = popdecs(r(1), :);
        x_r2 = popdecs(r(2), :);
        x_r3 = popdecs(r(3), :);
        x_r4 = popdecs(r(4), :);
        x_r5 = popdecs(r(5), :);
        
        % Differential components
        diff1 = F1(i) * (x_r1 - x_r2);
        diff2 = F2(i) * (x_r3 - x_r4);
        
        % Constraint-aware term
        cons_term = beta(i) * (c_min - cons(i)) * x_r5;
        
        % Combine components
        offspring(i,:) = x_best + diff1 + diff2 + cons_term;
        
        % Add best individual guidance with 30% probability
        if rand() < 0.3
            offspring(i,:) = offspring(i,:) + 0.15 * (x_best - popdecs(i,:));
        end
    end
    
    % Boundary handling with adaptive reflection
    lb = -1000;
    ub = 1000;
    for d = 1:D
        % Calculate dimension-wise diversity
        dim_diversity = pop_std(d) / (pop_std(d) + 100);
        reflect_factor = 0.3 + 0.4 * dim_diversity;
        
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