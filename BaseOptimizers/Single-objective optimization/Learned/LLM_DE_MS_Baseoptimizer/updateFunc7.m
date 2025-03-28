function [offspring] = updateFunc7(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Parameters
    lambda = 0.1;
    elite_ratio = 0.3;
    eps = 1e-12;
    
    % Normalize constraint violations
    c_max = max(abs(cons));
    c_norm = abs(cons) ./ max(c_max, eps);
    
    % Sort by fitness and get elite individuals
    [~, sorted_idx] = sort(popfits);
    elite_count = round(NP * elite_ratio);
    elite_idx = sorted_idx(1:elite_count);
    
    % Find best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    for i = 1:NP
        % Select elite individual
        elite = elite_idx(randi(elite_count));
        
        % Select four distinct random individuals
        candidates = setdiff(1:NP, [i, best_idx, elite]);
        r = candidates(randperm(length(candidates), 4));
        r1 = r(1); r2 = r(2); r3 = r(3); r4 = r(4);
        
        % Adaptive scaling factors
        F1 = 0.5 * (1 + popfits(best_idx)/(popfits(i) + eps);
        F2 = 0.3 * (1 - c_norm(i));
        
        % Mutation
        offspring(i,:) = x_best + ...
                        F1 * (popdecs(r1,:) - popdecs(r2,:)) + ...
                        F2 * (popdecs(r3,:) - popdecs(r4,:)) + ...
                        lambda * (1 - c_norm(i)) * (popdecs(elite,:) - popdecs(i,:));
    end
end