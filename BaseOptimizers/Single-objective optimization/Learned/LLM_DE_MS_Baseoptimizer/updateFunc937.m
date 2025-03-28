% MATLAB Code
function [offspring] = updateFunc937(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify best individual considering constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        temp_idx = find(feasible);
        x_best = popdecs(temp_idx(best_idx), :);
        f_avg = mean(popfits(feasible));
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx, :);
        f_avg = mean(popfits);
    end
    
    % 2. Normalize constraint violations
    c_norm = abs(cons) / (max(abs(cons)) + eps);
    
    % 3. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % 4. Adaptive scaling factor
    F = 0.5 + 0.3 * rand(NP, 1) .* (1 - c_norm);
    
    % 5. Novel mutation strategy
    fitness_ratio = f_avg ./ (abs(popfits) + eps);
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    mutants = x_best(ones(NP,1),:) + ...
              F .* (fitness_ratio .* diff1 + (1 - c_norm) .* diff2);
    
    % 6. Dynamic crossover rate
    CR = 0.7 + 0.2 * rand(NP, 1) .* (1 - c_norm);
    
    % 7. Crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Smart boundary handling
    out_of_bounds = (offspring < lb) | (offspring > ub);
    offspring(out_of_bounds) = popdecs(out_of_bounds) + ...
                               rand(sum(out_of_bounds(:)), 1) .* ...
                               (x_best(out_of_bounds) - popdecs(out_of_bounds));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end