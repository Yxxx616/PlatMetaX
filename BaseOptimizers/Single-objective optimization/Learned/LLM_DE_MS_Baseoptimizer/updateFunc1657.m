% MATLAB Code
function [offspring] = updateFunc1657(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints and fitness
    abs_cons = abs(cons);
    norm_cons = abs_cons / (max(abs_cons) + eps;
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
        F1 = 0.5 * (1 - norm_cons(i));
        F2 = 0.5 * norm_cons(i);
        F3 = 0.3 + 0.4 * rand();
        
        % Constraint-aware crossover rate
        CR = 0.9 - 0.5 * norm_cons(i);
        
        % Enhanced mutation strategy
        constraint_diff = (1 + norm_cons(r1) - norm_cons(r2));
        fitness_ratio = norm_fits(r4) / (norm_fits(r3) + norm_fits(r4) + eps);
        
        mutation = popdecs(i,:) + ...
                   F1 .* (x_best - popdecs(i,:)) + ...
                   F2 .* (popdecs(r1,:) - popdecs(r2,:)) .* constraint_diff + ...
                   F3 .* (popdecs(r3,:) - popdecs(r4,:)) .* fitness_ratio;
        
        % Binomial crossover
        j_rand = randi(D);
        mask = rand(1, D) < CR;
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
    end
    
    % Boundary handling with reflection
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    offspring(viol_low) = 2*lb(viol_low) - offspring(viol_low);
    offspring(viol_high) = 2*ub(viol_high) - offspring(viol_high);
    
    % Final clipping with random perturbation
    offspring = max(min(offspring, ub), lb);
    rand_mask = rand(NP,D) < 0.1;
    offspring(rand_mask) = lb(rand_mask) + rand(sum(rand_mask(:)),1).*(ub(rand_mask)-lb(rand_mask));
end