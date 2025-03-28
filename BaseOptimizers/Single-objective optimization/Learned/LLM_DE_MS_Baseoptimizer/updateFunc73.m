function [offspring] = updateFunc73(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Adaptive parameters
    F1 = 0.5 + 0.3 * rand(NP, 1);  % Random F between 0.5-0.8
    F2 = 0.3 + 0.2 * rand(NP, 1);  % Random F between 0.3-0.5
    alpha = 0.1 + 0.1 * rand(NP, 1); % Random alpha between 0.1-0.2
    
    % Find best individual considering both fitness and constraints
    [~, best_fit_idx] = min(popfits);
    [~, best_cons_idx] = min(cons);
    x_best_fit = popdecs(best_fit_idx, :);
    x_best_cons = popdecs(best_cons_idx, :);
    x_best = (x_best_fit + x_best_cons) / 2;
    
    % Get minimum constraint violation
    c_best = min(cons);
    
    % Calculate population diversity
    pop_mean = mean(popdecs, 1);
    diversity = mean(std(popdecs, 0, 1));
    diversity_factor = 1.0 + 0.5 * (1 - diversity/1000); % Scale to [1, 1.5]
    
    for i = 1:NP
        % Select 5 distinct random vectors (excluding current and best)
        candidates = setdiff(1:NP, [i, best_fit_idx, best_cons_idx]);
        r = candidates(randperm(length(candidates), 5));
        x_r1 = popdecs(r(1), :);
        x_r2 = popdecs(r(2), :);
        x_r3 = popdecs(r(3), :);
        x_r4 = popdecs(r(4), :);
        x_r5 = popdecs(r(5), :);
        
        % Differential components
        diff1 = (x_r1 - x_r2) * F1(i) * diversity_factor;
        diff2 = (x_r3 - x_r4) * F2(i) * diversity_factor;
        
        % Constraint-aware term
        cons_term = alpha(i) * (c_best - cons(i)) * x_r5;
        
        % Combine all components
        offspring(i,:) = x_best + diff1 + diff2 + cons_term;
        
        % Occasionally add best individual's influence
        if rand() < 0.3
            offspring(i,:) = offspring(i,:) + 0.1 * (x_best - popdecs(i,:));
        end
    end
    
    % Boundary control with reflection
    lb = -1000;
    ub = 1000;
    for d = 1:D
        % Find out-of-bounds indices
        below = offspring(:,d) < lb;
        above = offspring(:,d) > ub;
        
        % Reflect and shrink
        offspring(below,d) = lb + (lb - offspring(below,d)) * 0.5;
        offspring(above,d) = ub - (offspring(above,d) - ub) * 0.5;
        
        % Final clipping
        offspring(:,d) = min(max(offspring(:,d), lb), ub);
    end
end