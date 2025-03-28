function [offspring] = updateFunc51(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Sort population by fitness
    [~, sorted_idx] = sort(popfits);
    elite_idx = sorted_idx(1:ceil(0.1*NP));
    avg_idx = sorted_idx(ceil(0.1*NP)+1:ceil(0.7*NP));
    poor_idx = sorted_idx(ceil(0.7*NP)+1:end);
    
    % Parameters
    F = 0.8;
    alpha = 0.5;
    
    for i = 1:NP
        % Select random vectors from each group
        elite = popdecs(elite_idx(randi(length(elite_idx)), :);
        avg1 = popdecs(avg_idx(randi(length(avg_idx))), :);
        avg2 = popdecs(avg_idx(randi(length(avg_idx))), :);
        avg3 = popdecs(avg_idx(randi(length(avg_idx))), :);
        poor = popdecs(poor_idx(randi(length(poor_idx))), :);
        
        % Mutation
        v = elite + F*(avg1 - poor) + F*(avg2 - avg3);
        
        % Constraint influence
        cons_factor = 1 - alpha * tanh(abs(cons(i)));
        v = v .* cons_factor;
        
        % Boundary control
        lb = -100 * ones(1, D);
        ub = 100 * ones(1, D);
        v = min(max(v, lb), ub);
        
        offspring(i,:) = v;
    end
end