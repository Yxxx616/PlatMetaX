% MATLAB Code
function [offspring] = updateFunc1133(popdecs, popfits, cons)
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
    F = 0.4 + 0.4 * (popfits - f_min) ./ (f_max - f_min + eps_val);
    CR = 0.1 + 0.8 * (1 - c ./ (c_max + eps_val));
    alpha = tanh(c_mean); % Constraint balance factor
    
    % 3. Fitness direction
    weights = exp(-((popfits - f_min).^2));
    d_fit = sum((popdecs - x_best) .* weights, 1) / (sum(weights) + eps_val);
    
    % 4. Constraint direction
    if c_mean > eps_val
        x_mean = mean(popdecs, 1);
        d_cons = sum((popdecs - x_mean) .* c, 1) / (sum(c) + eps_val);
    else
        d_cons = zeros(1,D);
    end
    
    % 5. Diversity direction
    d_div = x_worst - x_best;
    
    % 6. Composite mutation
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    F_expanded = repmat(F, 1, D);
    dir_composite = alpha*d_fit + (1-alpha)*(0.5*d_cons + 0.5*d_div);
    mutants = x_best + F_expanded .* dir_composite + ...
              F_expanded .* (popdecs(r1,:) - popdecs(r2,:));
    
    % 7. Constraint-aware crossover
    mask = rand(NP,D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    offspring = min(max(offspring, lb), ub);
end