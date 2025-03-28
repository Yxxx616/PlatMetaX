% MATLAB Code
function [offspring] = updateFunc1131(popdecs, popfits, cons)
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
    c = max(0, cons);
    r_c = mean(c > 0);
    
    % 2. Adaptive parameters
    F = 0.4 + 0.4 * (popfits - f_min) ./ (f_max - f_min + eps_val);
    CR = 0.1 + 0.8 * (1 - c ./ (max(c) + eps_val));
    
    % 3. Direction vectors
    % Elite direction
    elite_mask = popfits < f_mean;
    if any(elite_mask)
        elite_w = exp(-((popfits(elite_mask) - f_mean).^2));
        d_elite = sum((popdecs(elite_mask,:) - x_best) .* elite_w, 1) / (sum(elite_w) + eps_val);
    else
        d_elite = zeros(1,D);
    end
    
    % Diversity direction
    d_div = x_worst - x_best;
    
    % Constraint direction
    if any(c > 0)
        x_mean = mean(popdecs, 1);
        d_cons = sum((popdecs - x_mean) .* c, 1) / (sum(c) + eps_val);
    else
        d_cons = zeros(1,D);
    end
    
    % 4. Mutation with adaptive directions
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    F_expanded = repmat(F, 1, D);
    mutants = x_best + F_expanded .* (d_elite + (1-r_c)*d_div + r_c*d_cons) + ...
              F_expanded .* (popdecs(r1,:) - popdecs(r2,:));
    
    % 5. Crossover with constraint awareness
    mask = rand(NP,D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 6. Boundary handling with midpoint reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = (lb(lb_mask) + popdecs(lb_mask)) / 2;
    offspring(ub_mask) = (ub(ub_mask) + popdecs(ub_mask)) / 2;
    offspring = min(max(offspring, lb), ub);
end