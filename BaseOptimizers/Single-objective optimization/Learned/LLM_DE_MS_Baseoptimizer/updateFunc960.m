% MATLAB Code
function [offspring] = updateFunc960(popdecs, popfits, cons)
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
    
    % 3. Generate random indices for differential vectors
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    invalid = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    while any(invalid)
        r1(invalid) = randi(NP, sum(invalid), 1);
        r2(invalid) = randi(NP, sum(invalid), 1);
        invalid = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    end
    
    % 4. Adaptive scaling factors based on constraints
    F = 0.5 * (1 + abs(cons)/(c_max + eps)).^0.5;
    
    % 5. Enhanced mutation with elite guidance and differential vectors
    mutants = centroid(ones(NP,1),:) + ...
              F.*(x_best(ones(NP,1),:) - popdecs) + ...
              0.5*(popdecs(r1,:) - popdecs(r2,:));
    
    % 6. Constraint-aware crossover rate
    CR = 0.9 * (1 - abs(cons)/(c_max + eps)).^0.25;
    
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