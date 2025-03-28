% MATLAB Code
function [offspring] = updateFunc529(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Calculate adaptive weights
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fit = (popfits - f_min) / (f_max - f_min + eps);
    
    abs_cons = abs(cons);
    c_max = max(abs_cons);
    norm_con = abs_cons / (c_max + eps);
    
    % Combined weight considering both fitness and constraints
    w = 0.7 * (1 - norm_fit) + 0.3 * (1 - norm_con);
    
    % Select elite individual (best feasible or least infeasible)
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(popfits + norm_con);
        elite = popdecs(elite_idx,:);
    end
    
    % Generate distinct random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i, 'stable')(randi(NP-1)), idx);
    r2 = arrayfun(@(i) setdiff(idx, i, 'stable')(randi(NP-1)), idx);
    r3 = arrayfun(@(i) setdiff(idx, i, 'stable')(randi(NP-1)), idx);
    r4 = arrayfun(@(i) setdiff(idx, i, 'stable')(randi(NP-1)), idx);
    
    % Calculate scaling factors
    F1 = 0.8 * w;
    F2 = 0.6 * (1 - w);
    F3 = 0.4 * (1 - w).^2;
    
    % Constraint adaptation factor
    alpha = 0.5;
    constraint_factor = 1 + alpha * norm_con;
    
    % Mutation operation
    diff1 = repmat(elite, NP, 1) - popdecs;
    diff2 = (popdecs(r1,:) - popdecs(r2,:)) .* repmat(constraint_factor, 1, D);
    diff3 = (popdecs(r3,:) - popdecs(r4,:)) .* randn(NP, D);
    
    offspring = popdecs + repmat(F1, 1, D) .* diff1 + ...
                repmat(F2, 1, D) .* diff2 + ...
                repmat(F3, 1, D) .* diff3;
    
    % Adaptive boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Calculate adaptive reflection factor based on weights
    reflect_factor = 0.5 + 0.5 * w;
    
    % Handle lower bound violations
    below = offspring < lb_rep;
    reflected_low = lb_rep + reflect_factor .* (lb_rep - offspring);
    offspring = offspring .* ~below + reflected_low .* below;
    
    % Handle upper bound violations
    above = offspring > ub_rep;
    reflected_high = ub_rep - reflect_factor .* (offspring - ub_rep);
    offspring = offspring .* ~above + reflected_high .* above;
    
    % Final projection with small random perturbation
    perturb = (rand(NP,D)-0.5).*0.05.*(ub_rep-lb_rep).*repmat(1-w,1,D);
    offspring = max(min(offspring + perturb, ub_rep), lb_rep);
end