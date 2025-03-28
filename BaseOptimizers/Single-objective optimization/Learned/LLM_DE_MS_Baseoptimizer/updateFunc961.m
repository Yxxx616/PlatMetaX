% MATLAB Code
function [offspring] = updateFunc961(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute weighted centroid
    f_mean = mean(popfits);
    f_std = std(popfits) + eps;
    c_max = max(abs(cons)) + eps;
    
    weights = 1./(1 + max(0, cons) + abs(popfits - f_mean)/f_std);
    weights = weights / sum(weights);
    centroid = weights' * popdecs;
    
    % 2. Identify best solution considering constraints
    [~, best_idx] = min(popfits + 100*max(0, cons));
    x_best = popdecs(best_idx,:);
    
    % 3. Generate four distinct random indices for each target
    r = zeros(NP, 4);
    for i = 1:NP
        available = setdiff(1:NP, i);
        r(i,:) = available(randperm(length(available), 4));
    end
    
    % 4. Adaptive scaling factors based on constraints
    F = 0.4 + 0.4 * (1 - abs(cons)/(c_max + eps));
    
    % 5. Enhanced mutation with multiple differential vectors
    mutants = centroid(ones(NP,1),:) + ...
              F.*(x_best(ones(NP,1),:) - popdecs) + ...
              0.6*(popdecs(r(:,1),:) - popdecs(r(:,2),:)) + ...
              0.4*(popdecs(r(:,3),:) - popdecs(r(:,4),:));
    
    % 6. Constraint-aware crossover rate
    CR = 0.85 * (1 - max(0, cons)/(c_max + eps)).^0.2;
    
    % 7. Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Adaptive boundary handling
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    % For lower bound violations
    below_lb = offspring < lb_matrix;
    offspring(below_lb) = (popdecs(below_lb) + lb_matrix(below_lb))/2 + ...
                         rand(sum(below_lb(:)),1) .* (ub_matrix(below_lb) - lb_matrix(below_lb))/10;
    
    % For upper bound violations
    above_ub = offspring > ub_matrix;
    offspring(above_ub) = (popdecs(above_ub) + ub_matrix(above_ub))/2 - ...
                         rand(sum(above_ub(:)),1) .* (ub_matrix(above_ub) - lb_matrix(above_ub))/10;
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_matrix), ub_matrix);
end