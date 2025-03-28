% MATLAB Code
function [offspring] = updateFunc1654(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    abs_cons = abs(cons);
    norm_cons = abs_cons / (max(abs_cons) + eps);
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    
    % Find best solution (considering both fitness and constraints)
    [~, best_idx] = min(popfits + 10*norm_cons); % Strong penalty for constraints
    x_best = popdecs(best_idx, :);
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    for i = 1:NP
        % Select 4 distinct random indices different from i
        candidates = setdiff(1:NP, i);
        r_idx = candidates(randperm(length(candidates), 4));
        r1 = r_idx(1); r2 = r_idx(2); r3 = r_idx(3); r4 = r_idx(4);
        
        % Adaptive parameters
        F1 = 0.5 * (1 + norm_cons(i)); % More exploration for constrained solutions
        F2 = 0.7 * (1 - norm_cons(i)); % Less perturbation for feasible solutions
        F3 = 0.3 + 0.4 * rand();
        
        % Constraint-aware crossover rate
        CR = 0.8 - 0.3 * norm_cons(i);
        
        % Novel mutation strategy
        constraint_diff = (1 + norm_cons(r1) - norm_cons(r2));
        fitness_ratio = norm_fits(r4) / (norm_fits(r3) + norm_fits(r4) + eps);
        
        mutation = popdecs(i,:) + ...
                   F1 * (x_best - popdecs(i,:)) + ...
                   F2 * (popdecs(r1,:) - popdecs(r2,:)) .* constraint_diff + ...
                   F3 * (popdecs(r3,:) - popdecs(r4,:)) .* fitness_ratio;
        
        % Exponential crossover
        j_rand = randi(D);
        L = 0;
        while rand() < CR && L < D
            L = L + 1;
        end
        indices = mod((j_rand:j_rand+L-1)-1, D) + 1;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,indices) = mutation(indices);
    end
    
    % Boundary handling with random reinitialization
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    offspring(viol_low) = lb(viol_low) + rand(sum(viol_low(:)),1) .* (ub(viol_low) - lb(viol_low));
    offspring(viol_high) = lb(viol_high) + rand(sum(viol_high(:)),1) .* (ub(viol_high) - lb(viol_high));
end