function [offspring] = updateFunc36(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Sort population by fitness
    [~, sorted_idx] = sort(popfits);
    
    % Determine indices for different fitness levels
    best_idx = sorted_idx(1:floor(0.1*NP));
    mid_idx = sorted_idx(floor(0.2*NP):floor(0.8*NP));
    worst_idx = sorted_idx(floor(0.9*NP):end);
    
    for i = 1:NP
        % Randomly select distinct vectors
        x_best = popdecs(best_idx(randi(length(best_idx))), :);
        x_mid = popdecs(mid_idx(randi(length(mid_idx))), :);
        x_worst = popdecs(worst_idx(randi(length(worst_idx))), :);
        
        % Calculate scaling factors
        F1 = 0.5 * (1 + rand());
        F2 = 0.3 * (1 - rand());
        
        % Calculate constraint direction
        cons_norm = norm(cons(i,:));
        if cons_norm > 0
            cons_dir = cons(i,:) / cons_norm;
        else
            cons_dir = zeros(1,D);
        end
        
        % Calculate difference vector norm
        diff_norm = norm(x_mid - x_worst);
        
        % Generate mutant vector
        offspring(i,:) = x_best + F1*(x_mid - x_worst) + ...
                        F2*cons_dir*diff_norm;
    end
    
    % Ensure bounds are maintained (optional)
    % offspring = min(max(offspring, lb), ub);
end