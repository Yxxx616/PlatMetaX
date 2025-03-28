function [offspring] = updateFunc37(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Sort population by fitness
    [~, sorted_idx] = sort(popfits);
    
    % Determine indices for different fitness levels
    best_idx = sorted_idx(1:max(1,floor(0.1*NP)));
    mid_idx = sorted_idx(max(1,floor(0.2*NP)):min(NP,floor(0.8*NP)));
    
    for i = 1:NP
        % Select distinct random vectors from different groups
        x_best = popdecs(best_idx(randi(length(best_idx))), :);
        
        % Select two distinct vectors from middle group
        candidates = mid_idx(randperm(length(mid_idx), 2));
        x_r1 = popdecs(candidates(1), :);
        x_r2 = popdecs(candidates(2), :);
        
        % Calculate scaling factors
        F1 = 0.5 * (1 + rand());
        F2 = 0.3 * (1 - rand());
        
        % Calculate constraint direction
        cons_norm = sqrt(sum(cons(i,:).^2));
        if cons_norm > 0
            cons_dir = cons(i,:) / cons_norm;
        else
            cons_dir = zeros(1,D);
        end
        
        % Calculate difference vector norm
        diff_norm = sqrt(sum((x_r1 - x_r2).^2));
        
        % Generate mutant vector
        offspring(i,:) = x_best + F1*(x_r1 - x_r2) + ...
                        F2*cons_dir*diff_norm;
    end
end