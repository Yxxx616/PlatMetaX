function [offspring] = updateFunc178(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Find feasible and infeasible solutions
    feasible_mask = cons <= 0;
    infeasible_mask = ~feasible_mask;
    feasible_pop = popdecs(feasible_mask, :);
    feasible_fits = popfits(feasible_mask);
    
    % Normalize fitness and constraints
    max_fit = max(abs(popfits));
    min_fit = min(popfits);
    max_cons = max(abs(cons));
    norm_fits = (popfits - min_fit) ./ (max_fit - min_fit + eps);
    norm_cons = abs(cons) ./ (max_cons + eps);
    
    % Find elite solution (best feasible if exists, otherwise best overall)
    if ~isempty(feasible_pop)
        [~, elite_idx] = max(feasible_fits);
        elite = feasible_pop(elite_idx, :);
    else
        [~, elite_idx] = max(popfits);
        elite = popdecs(elite_idx, :);
    end
    
    for i = 1:NP
        % Adaptive parameters
        F1 = 0.8 * (1 - norm_cons(i));
        F2 = 0.5 * (1 + norm_fits(i));
        F3 = 0.3 * norm_cons(i);
        
        % Select random vectors
        r1 = randi(NP);
        r2 = randi(NP);
        while r2 == r1
            r2 = randi(NP);
        end
        
        % Select feasible and infeasible vectors
        if any(feasible_mask) && any(infeasible_mask)
            feasible_idx = randi(sum(feasible_mask));
            infeasible_idx = randi(sum(infeasible_mask));
            feasible_vec = feasible_pop(feasible_idx, :);
            infeasible_vec = popdecs(infeasible_mask, :);
            infeasible_vec = infeasible_vec(infeasible_idx, :);
        else
            feasible_vec = elite;
            infeasible_vec = popdecs(i, :);
        end
        
        % Mutation
        w = norm_fits(i);
        c = norm_cons(i);
        diff1 = elite - popdecs(i, :);
        diff2 = popdecs(r1, :) - popdecs(r2, :);
        diff3 = feasible_vec - infeasible_vec;
        
        offspring(i, :) = popdecs(i, :) + F1*diff1 + F2*w*diff2 + F3*c*diff3;
    end
    
    % Boundary control
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = min(max(offspring, lb), ub);
end