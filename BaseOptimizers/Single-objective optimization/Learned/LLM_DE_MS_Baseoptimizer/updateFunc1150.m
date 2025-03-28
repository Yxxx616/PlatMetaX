% MATLAB Code
function [offspring] = updateFunc1150(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps_val = 1e-10;
    
    % 1. Population analysis
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    % Feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask)/NP;
    if any(feasible_mask)
        x_feas = mean(popdecs(feasible_mask, :), 1);
    else
        x_feas = x_best;
    end
    if any(~feasible_mask)
        x_infeas = mean(popdecs(~feasible_mask, :), 1);
    else
        x_infeas = x_worst;
    end
    
    % 2. Compute direction vectors
    d_feas = x_feas - x_infeas;
    d_fit = x_best - x_worst;
    
    % 3. Adaptive weights
    alpha = 0.5 + 0.4*tanh(10*(rho - 0.5));
    
    % 4. Generate random indices for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2 = randi(NP, NP, 1);
    end
    d_div = popdecs(r1,:) - popdecs(r2,:);
    
    % 5. Adaptive scaling factor
    c_norm = (cons - min(cons)) ./ (max(cons) - min(cons) + eps_val);
    F_base = 0.5;
    F = F_base + 0.3 * (1 - c_norm);
    
    % 6. Mutation
    mutants = popdecs + repmat(F, 1, D) .* (repmat(d_feas, NP, 1) + ...
              alpha * repmat(d_fit, NP, 1) + ...
              (1-alpha) * d_div);
    
    % 7. Dynamic crossover
    CR_base = 0.9;
    CR = CR_base - 0.4 * c_norm;
    mask = rand(NP,D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with bounce-back
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = lb(lb_mask) + rand(sum(lb_mask(:)),1) .* ...
                         (popdecs(lb_mask) - lb(lb_mask));
    offspring(ub_mask) = ub(ub_mask) - rand(sum(ub_mask(:)),1) .* ...
                         (ub(ub_mask) - popdecs(ub_mask));
    offspring = min(max(offspring, lb), ub);
end