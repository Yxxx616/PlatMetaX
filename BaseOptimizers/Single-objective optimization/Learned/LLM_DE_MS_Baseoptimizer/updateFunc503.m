% MATLAB Code
function [offspring] = updateFunc503(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-10;
    
    % Identify elite solution
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(abs(cons));
        elite = popdecs(elite_idx, :);
    end
    
    % Compute adaptive weights
    min_fit = min(popfits);
    max_fit = max(popfits);
    w_f = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    abs_cons = abs(cons);
    max_con = max(abs_cons);
    w_c = abs_cons / (max_con + eps);
    
    % Generate random indices
    [~, idx] = sort(rand(NP, NP), 2);
    r1 = idx(:,1);
    r2 = idx(:,2);
    r3 = idx(:,3);
    
    % Ensure indices are different
    mask = r1 == (1:NP)'; r1(mask) = r2(mask);
    mask = r2 == (1:NP)'; r2(mask) = r3(mask);
    mask = r3 == (1:NP)'; r3(mask) = idx(mask,4);
    
    % Adaptive scaling factors
    F_base = 0.5 + 0.3 * rand(NP,1);
    F_fit = 0.3 * (1 - w_f) .* rand(NP,1);
    F_con = 0.2 * w_c .* rand(NP,1);
    rand_pert = 0.1 * randn(NP,D) .* (1 - w_f) .* (1 - w_c);
    
    % Mutation components
    elite_dir = repmat(elite, NP, 1) - popdecs;
    fit_diff = popdecs(r1,:) - popdecs(r2,:);
    con_diff = popdecs(r3,:) - popdecs;
    
    % Generate offspring
    offspring = popdecs + repmat(F_base,1,D).*elite_dir + ...
                repmat(F_fit,1,D).*fit_diff + ...
                repmat(F_con,1,D).*con_diff + ...
                rand_pert;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    offspring = offspring.*(~below & ~above) + ...
                (2*lb_rep - offspring).*below + ...
                (2*ub_rep - offspring).*above;
    
    % Final bounds check
    offspring = max(min(offspring, ub_rep), lb_rep);
end