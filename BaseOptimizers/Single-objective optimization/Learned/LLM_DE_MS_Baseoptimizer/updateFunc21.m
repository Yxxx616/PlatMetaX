function [offspring] = updateFunc21(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Sort population by fitness
    [~, sorted_idx] = sort(popfits);
    sorted_pop = popdecs(sorted_idx, :);
    
    % Determine segments
    n_best = max(1, round(NP*0.2));
    n_mid = max(1, round(NP*0.6));
    
    % Get reference vectors
    best_pop = sorted_pop(1:n_best, :);
    mid_pop = sorted_pop(n_best+1:n_best+n_mid, :);
    worst_pop = sorted_pop(end-n_best+1:end, :);
    
    % Normalize constraints and fitness
    max_cons = max(abs(cons));
    max_fit = max(abs(popfits));
    
    for i = 1:NP
        % Select random vectors from each segment
        r1 = randi(size(best_pop, 1));
        r2 = randi(size(mid_pop, 1));
        r3 = randi(size(worst_pop, 1));
        
        % Calculate adaptive factors
        F1 = 0.5 * (1 + cons(i)/max_cons);
        F2 = 0.5 * (1 - popfits(i)/max_fit);
        
        % Mutation
        offspring(i,:) = best_pop(r1,:) + ...
                        F1 * (mid_pop(r2,:) - worst_pop(r3,:)) + ...
                        F2 * (best_pop(r1,:) - popdecs(i,:));
        
        % Boundary control
        lb = -600 * ones(1, D);
        ub = 600 * ones(1, D);
        for j = 1:D
            if offspring(i,j) < lb(j)
                offspring(i,j) = (popdecs(i,j) + lb(j))/2;
            elseif offspring(i,j) > ub(j)
                offspring(i,j) = (popdecs(i,j) + ub(j))/2;
            end
        end
    end
end