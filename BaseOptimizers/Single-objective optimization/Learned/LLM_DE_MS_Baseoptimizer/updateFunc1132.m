% MATLAB Code
function [offspring] = updateFunc1132(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps_val = 1e-10;
    
    % 1. Population analysis
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    f_mean = mean(popfits);
    f_min = min(popfits);
    f_max = max(popfits);
    c = max(0, cons); % Ensure non-negative constraints
    c_mean = mean(c);
    c_max = max(c);
    
    % 2. Adaptive parameters
    F = 0.5 * (1 + (popfits - f_min) ./ (f_max - f_min + eps_val));
    CR = 0.1 + 0.8 * (1 - c ./ (c_max + eps_val));
    alpha = c_mean / (c_max + eps_val);
    
    % 3. Fitness-guided direction
    weights = exp(-((popfits - f_min).^2));
    d_fit = sum((popdecs - x_best) .* weights, 1) / (sum(weights) + eps_val);
    
    % 4. Constraint direction
    if c_mean > 0
        x_mean = mean(popdecs, 1);
        d_cons = sum((popdecs - x_mean) .* c, 1) / (sum(c) + eps_val);
    else
        d_cons = zeros(1,D);
    end
    
    % 5. Diversity direction
    d_div = x_worst - x_best;
    
    % 6. Mutation with adaptive directions
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    F_expanded = repmat(F, 1, D);
    mutants = x_best + F_expanded .* (d_fit + alpha*d_cons + (1-alpha)*d_div) + ...
              F_expanded .* (popdecs(r1,:) - popdecs(r2,:));
    
    % 7. Crossover with constraint awareness
    mask = rand(NP,D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = lb(lb_mask) + abs(lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = ub(ub_mask) - abs(offspring(ub_mask) - ub(ub_mask));
    offspring = min(max(offspring, lb), ub);
end