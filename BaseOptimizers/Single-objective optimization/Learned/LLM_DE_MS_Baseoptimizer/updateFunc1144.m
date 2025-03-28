% MATLAB Code
function [offspring] = updateFunc1144(popdecs, popfits, cons)
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
    
    % Separate feasible and infeasible solutions
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        x_feas_mean = mean(popdecs(feasible_mask, :), 1);
    else
        x_feas_mean = x_mean;
    end
    if any(~feasible_mask)
        x_infeas_mean = mean(popdecs(~feasible_mask, :), 1);
    else
        x_infeas_mean = x_mean;
    end
    
    f_mean = mean(popfits);
    f_min = min(popfits);
    f_max = max(popfits);
    c = max(0, cons); % Ensure non-negative constraints
    c_mean = mean(c);
    c_max = max(c);
    c_ratio = sum(feasible_mask)/NP;
    
    % 2. Direction vectors
    d_fit = x_best - x_mean;
    d_div = x_worst - x_mean;
    d_cons = x_feas_mean - x_infeas_mean;
    
    % 3. Adaptive combination weights
    alpha = 1 ./ (1 + exp(-10*(c_ratio-0.3)));
    beta = 0.7*(1 - c_ratio);
    d_composite = alpha.*d_fit + (1-alpha).*(beta.*d_cons + (1-beta).*d_div);
    
    % 4. Mutation with enhanced exploration
    F = 0.4 + 0.4*(popfits - f_min)./(f_max - f_min + eps_val);
    F_expanded = repmat(F, 1, D);
    
    % Select three distinct random vectors
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2 = randi(NP, NP, 1);
    end
    r3 = randi(NP, NP, 1);
    while any(r3 == r1 | r3 == r2)
        r3 = randi(NP, NP, 1);
    end
    
    mutants = popdecs + F_expanded .* d_composite + ...
              F_expanded .* (popdecs(r1,:) - popdecs(r2,:)) + ...
              0.7*F_expanded .* (popdecs(r3,:) - repmat(x_mean, NP, 1));
    
    % 5. Dynamic crossover
    CR = 0.7 + 0.2*(1 - c./(c_max + eps_val));
    mask = rand(NP,D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 6. Boundary handling with reflection and clamping
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    offspring = min(max(offspring, lb), ub);
end