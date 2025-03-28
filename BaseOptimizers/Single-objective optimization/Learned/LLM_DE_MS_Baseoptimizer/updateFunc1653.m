% MATLAB Code
function [offspring] = updateFunc1653(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    abs_cons = abs(cons);
    norm_cons = abs_cons / (max(abs_cons) + eps);
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    
    % Find best solution
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    for i = 1:NP
        % Select 4 distinct random indices different from i
        candidates = setdiff(1:NP, i);
        r_idx = candidates(randperm(length(candidates), 4));
        r1 = r_idx(1); r2 = r_idx(2); r3 = r_idx(3); r4 = r_idx(4);
        
        % Adaptive parameters
        F = 0.5 + 0.3 * randn();
        CR = 0.9 - 0.2 * norm_cons(i);
        
        % Constraint-aware mutation
        constraint_factor = 1 + norm_cons(i);
        fitness_weight = norm_fits(r3) / (norm_fits(r3) + norm_fits(r4) + eps);
        
        mutation = popdecs(i,:) + F * (...
            (x_best - popdecs(i,:)) + ...
            constraint_factor * (popdecs(r1,:) - popdecs(r2,:)) + ...
            fitness_weight * (popdecs(r3,:) - popdecs(r4,:)));
        
        % Crossover
        mask = rand(1,D) < CR;
        j_rand = randi(D);
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:).*(~mask) + mutation.*mask;
    end
    
    % Boundary handling with reflection
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    offspring(viol_low) = min(ub(viol_low), 2*lb(viol_low) - offspring(viol_low));
    offspring(viol_high) = max(lb(viol_high), 2*ub(viol_high) - offspring(viol_high));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end