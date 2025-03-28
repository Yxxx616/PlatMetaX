function [offspring] = updateFunc22(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Sort population by fitness (ascending order for minimization)
    [~, sorted_idx] = sort(popfits);
    sorted_pop = popdecs(sorted_idx, :);
    
    % Determine segments
    n_elite = max(1, round(NP*0.2));
    n_medium = max(1, round(NP*0.6));
    
    elite_pop = sorted_pop(1:n_elite, :);
    medium_pop = sorted_pop(n_elite+1:n_elite+n_medium, :);
    poor_pop = sorted_pop(end-n_elite+1:end, :);
    
    % Normalize constraints and fitness
    max_cons = max(abs(cons));
    min_fit = min(popfits);
    max_fit = max(popfits);
    range_fit = max_fit - min_fit;
    
    for i = 1:NP
        % Select random vectors from each segment
        r_elite = randi(size(elite_pop, 1));
        r_medium = randi(size(medium_pop, 1));
        r_poor = randi(size(poor_pop, 1));
        
        % Calculate adaptive factors
        F = 0.5 + 0.5 * rand() * (1 - abs(cons(i))/max_cons);
        alpha = 0.5 * (1 + (popfits(i) - min_fit)/range_fit);
        
        % Mutation
        offspring(i,:) = elite_pop(r_elite,:) + ...
                        F * (medium_pop(r_medium,:) - poor_pop(r_poor,:)) + ...
                        alpha * (elite_pop(r_elite,:) - popdecs(i,:));
        
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