% MATLAB Code
function [offspring] = updateFunc1658(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    abs_cons = abs(cons);
    norm_cons = abs_cons / (max(abs_cons) + eps);
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    
    % Find best solution considering both fitness and constraints
    combined_score = popfits + 100*norm_cons;
    [~, best_idx] = min(combined_score);
    x_best = popdecs(best_idx, :);
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    for i = 1:NP
        % Select 4 distinct random indices different from i
        candidates = setdiff(1:NP, i);
        r_idx = candidates(randperm(length(candidates), 4));
        r1 = r_idx(1); r2 = r_idx(2); r3 = r_idx(3); r4 = r_idx(4);
        
        % Adaptive scaling factors
        F1 = 0.7 * (1 - norm_cons(i));  % Stronger guidance for feasible solutions
        F2 = 0.5 * (1 + norm_cons(i));  % Increased exploration for infeasible
        F3 = 0.3 + 0.3 * rand();        % Random component
        
        % Constraint-aware crossover rate
        CR = 0.85 - 0.4 * norm_cons(i);
        
        % Constraint weights
        w_con = 1 - norm_cons(i);
        fitness_ratio = popfits(r2) / (popfits(r1) + popfits(r2) + eps);
        constraint_ratio = 1 + norm_cons(r3) / (norm_cons(r4) + eps);
        
        % Enhanced mutation strategy
        mutation = popdecs(i,:) + ...
                  F1 .* (x_best - popdecs(i,:)) .* w_con + ...
                  F2 .* (popdecs(r1,:) - popdecs(r2,:)) .* fitness_ratio + ...
                  F3 .* (popdecs(r3,:) - popdecs(r4,:)) .* constraint_ratio;
        
        % Binomial crossover
        j_rand = randi(D);
        mask = rand(1, D) < CR;
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
    end
    
    % Boundary handling with adaptive reflection
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    offspring(viol_low) = lb(viol_low) + rand(sum(viol_low(:)),1) .* (x_best(viol_low) - lb(viol_low));
    offspring(viol_high) = ub(viol_high) - rand(sum(viol_high(:)),1) .* (ub(viol_high) - x_best(viol_high));
    
    % Final clipping with probability-based perturbation
    rand_mask = rand(NP,D) < 0.05;
    offspring(rand_mask) = lb(rand_mask) + rand(sum(rand_mask(:)),1).*(ub(rand_mask)-lb(rand_mask));
end