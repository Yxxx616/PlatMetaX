function [offspring] = updateFunc177(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Sort population based on fitness
    [~, sorted_idx] = sort(popfits, 'descend');
    sorted_pop = popdecs(sorted_idx, :);
    
    % Classify population into groups
    elite_num = ceil(0.2 * NP);
    avg_num = ceil(0.6 * NP);
    poor_num = NP - elite_num - avg_num;
    
    elite_pop = sorted_pop(1:elite_num, :);
    avg_pop = sorted_pop(elite_num+1:elite_num+avg_num, :);
    poor_pop = sorted_pop(elite_num+avg_num+1:end, :);
    
    % Normalize constraints and fitness
    max_cons = max(abs(cons));
    max_fit = max(abs(popfits));
    norm_cons = cons ./ max_cons;
    norm_fits = popfits ./ max_fit;
    
    for i = 1:NP
        % Select random vectors from each group
        r1 = randi(size(elite_pop, 1));
        r2 = randi(size(avg_pop, 1), [1, 2]);
        r3 = randi(size(avg_pop, 1));
        r4 = randi(size(poor_pop, 1));
        
        % Adaptive parameters
        F = 0.5 * (1 + norm_cons(i));
        lambda = 0.5 * (1 - norm_fits(i));
        
        % Mutation
        diff1 = avg_pop(r2(1), :) - avg_pop(r2(2), :);
        diff2 = poor_pop(r4, :) - avg_pop(r3, :);
        offspring(i, :) = elite_pop(r1, :) + F * diff1 + lambda * diff2;
    end
    
    % Boundary control
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = min(max(offspring, lb), ub);
end