% MATLAB Code
function [offspring] = updateFunc964(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute constraint-weighted centroid
    f_mean = mean(popfits);
    f_std = std(popfits) + eps;
    c_max = max(abs(cons)) + eps;
    
    weights = 1./(1 + max(0, cons) + abs(popfits - f_mean)/f_std);
    weights = weights / sum(weights);
    centroid = weights' * popdecs;
    
    % 2. Identify elite solution (best feasible or least infeasible)
    [~, elite_idx] = min(popfits + 100*max(0, cons));
    x_elite = popdecs(elite_idx,:);
    
    % 3. Generate four distinct random indices for each target
    r = zeros(NP, 4);
    for i = 1:NP
        available = setdiff(1:NP, i);
        r(i,:) = available(randperm(length(available), 4));
    end
    
    % 4. Adaptive scaling factors based on constraints
    F = 0.5 * (1 - abs(cons)/(c_max + eps)).^0.5;
    
    % 5. Enhanced hybrid mutation
    mutants = centroid(ones(NP,1),:) + ...
              F.*(x_elite(ones(NP,1),:) - popdecs) + ...
              0.6*(popdecs(r(:,1),:) - popdecs(r(:,2),:)) + ...
              0.4*(popdecs(r(:,3),:) - popdecs(r(:,4),:));
    
    % 6. Dynamic crossover rate
    CR = 0.9 * (1 - max(0, cons)/(c_max + eps)).^0.2;
    
    % 7. Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Improved boundary handling with reflection
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    % Lower bound violations
    below_lb = offspring < lb_matrix;
    rand_vals = rand(sum(below_lb(:)),1);
    offspring(below_lb) = lb_matrix(below_lb) + abs(offspring(below_lb) - lb_matrix(below_lb)) .* (1 + 0.1*rand_vals);
    
    % Upper bound violations
    above_ub = offspring > ub_matrix;
    rand_vals = rand(sum(above_ub(:)),1);
    offspring(above_ub) = ub_matrix(above_ub) - abs(offspring(above_ub) - ub_matrix(above_ub)) .* (1 + 0.1*rand_vals);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_matrix), ub_matrix);
end