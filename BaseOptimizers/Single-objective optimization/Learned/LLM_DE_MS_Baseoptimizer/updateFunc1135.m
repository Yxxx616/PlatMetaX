% MATLAB Code
function [offspring] = updateFunc1135(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps_val = 1e-10;
    
    % 1. Population analysis
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    x_mean = mean(popdecs, 1);
    f_mean = mean(popfits);
    f_min = min(popfits);
    f_max = max(popfits);
    c = max(0, cons); % Ensure non-negative constraints
    c_mean = mean(c);
    c_max = max(c);
    
    % 2. Direction vectors
    d_fit = x_best - x_mean;
    if c_mean > eps_val
        weighted_mean = sum(popdecs .* c, 1) / (sum(c) + eps_val);
        d_cons = weighted_mean - x_mean;
    else
        d_cons = zeros(1,D);
    end
    d_div = x_worst - x_best;
    
    % 3. Adaptive combination
    alpha = 1 ./ (1 + exp(-10*(c_mean-0.5)));
    d_composite = alpha*d_fit + (1-alpha)*(0.6*d_cons + 0.4*d_div);
    
    % 4. Mutation
    F = 0.3 + 0.5*(popfits - f_min)./(f_max - f_min + eps_val);
    F_expanded = repmat(F, 1, D);
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2 = randi(NP, NP, 1);
    end
    mutants = popdecs + F_expanded .* d_composite + ...
              F_expanded .* (popdecs(r1,:) - popdecs(r2,:));
    
    % 5. Crossover
    CR = 0.5 + 0.4*(1 - c./(c_max + eps_val));
    mask = rand(NP,D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    offspring = min(max(offspring, lb), ub);
end