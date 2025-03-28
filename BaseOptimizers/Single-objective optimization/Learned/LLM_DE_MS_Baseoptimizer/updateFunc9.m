function [offspring] = updateFunc9(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Parameters
    eps = 1e-12;
    elite_ratio = 0.2;
    
    % Normalize constraint violations
    c_max = max(abs(cons));
    c_norm = abs(cons) ./ max(c_max, eps);
    
    % Sort by fitness and get elite individuals
    [~, sorted_idx] = sort(popfits);
    elite_count = max(1, round(NP * elite_ratio));
    elite_idx = sorted_idx(1:elite_count);
    
    % Find best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Calculate ranks (1 is best)
    ranks = zeros(NP, 1);
    [~, rank_order] = sort(popfits);
    ranks(rank_order) = 1:NP;
    
    for i = 1:NP
        % Select elite individual
        elite = elite_idx(randi(elite_count));
        
        % Select four distinct random individuals
        candidates = setdiff(1:NP, [i, best_idx, elite]);
        r = candidates(randperm(length(candidates), min(4, length(candidates))));
        r1 = r(1); 
        r2 = r(2);
        if length(r) >= 4
            r3 = r(3); 
            r4 = r(4);
        else
            r3 = r(1); 
            r4 = r(2);
        end
        
        % Adaptive scaling factors
        F1 = 0.6 * (1 - c_norm(i));
        F2 = 0.4 * (popfits(best_idx)/(popfits(i) + eps);
        F3 = 0.2 * (1 - ranks(i)/NP);
        
        % Mutation
        offspring(i,:) = x_best + ...
                        F1 * (popdecs(r1,:) - popdecs(r2,:)) + ...
                        F2 * (popdecs(r3,:) - popdecs(r4,:)) + ...
                        F3 * (popdecs(elite,:) - popdecs(i,:));
    end
end