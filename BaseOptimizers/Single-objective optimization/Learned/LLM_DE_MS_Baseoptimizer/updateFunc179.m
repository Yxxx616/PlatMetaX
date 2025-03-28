function [offspring] = updateFunc179(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Identify feasible and infeasible solutions
    feasible_mask = cons <= 0;
    infeasible_mask = ~feasible_mask;
    feasible_pop = popdecs(feasible_mask, :);
    feasible_fits = popfits(feasible_mask);
    
    % Normalize fitness and constraints
    norm_fits = (popfits - min(popfits)) ./ (max(popfits) - min(popfits) + eps;
    norm_cons = abs(cons) ./ (max(abs(cons)) + eps;
    
    % Determine elite solution
    if ~isempty(feasible_pop)
        [~, elite_idx] = max(feasible_fits);
        elite = feasible_pop(elite_idx, :);
    else
        [~, elite_idx] = max(popfits);
        elite = popdecs(elite_idx, :);
    end
    
    % Pre-compute random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    same_idx = (r1 == r2);
    r2(same_idx) = mod(r2(same_idx) + randi(NP-1), NP) + 1;
    
    % Vectorized mutation
    for i = 1:NP
        % Adaptive parameters
        F1 = 0.9 * (1 - norm_cons(i));
        F2 = 0.5 * (1 + norm_fits(i));
        F3 = 0.3 * norm_cons(i);
        
        % Select feasible and infeasible vectors
        if any(feasible_mask) && any(infeasible_mask)
            feasible_idx = randi(sum(feasible_mask));
            infeasible_idx = randi(sum(infeasible_mask));
            feas_vec = feasible_pop(feasible_idx, :);
            infeas_vec = popdecs(infeasible_mask, :);
            infeas_vec = infeas_vec(infeasible_idx, :);
        else
            feas_vec = elite;
            infeas_vec = popdecs(i, :);
        end
        
        % Mutation components
        elite_diff = elite - popdecs(i, :);
        feas_diff = feas_vec - infeas_vec;
        rand_diff = popdecs(r1(i), :) - popdecs(r2(i), :);
        
        % Combined mutation
        offspring(i, :) = popdecs(i, :) + F1*elite_diff + F2*feas_diff + F3*rand_diff;
    end
    
    % Boundary control
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = min(max(offspring, lb), ub);
end