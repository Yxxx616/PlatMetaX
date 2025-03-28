% MATLAB Code
function [offspring] = updateFunc504(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-10;
    
    % Identify elite solution considering both fitness and constraints
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        elite = popdecs(temp(elite_idx), :);
    else
        % Combine normalized fitness and constraints for infeasible solutions
        norm_fit = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
        norm_con = abs(cons) / (max(abs(cons)) + eps);
        combined = 0.7*norm_con + 0.3*norm_fit;
        [~, elite_idx] = min(combined);
        elite = popdecs(elite_idx, :);
    end
    
    % Compute adaptive weights with enhanced normalization
    min_fit = min(popfits);
    max_fit = max(popfits);
    w_f = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    abs_cons = abs(cons);
    max_con = max(abs_cons);
    w_c = abs_cons / (max_con + eps);
    
    % Generate unique random indices
    [~, idx] = sort(rand(NP, NP), 2);
    r1 = idx(:,1);
    r2 = idx(:,2);
    r3 = idx(:,3);
    
    % Ensure indices are different from current index
    mask = r1 == (1:NP)'; r1(mask) = idx(mask,4);
    mask = r2 == (1:NP)'; r2(mask) = idx(mask,5);
    mask = r3 == (1:NP)'; r3(mask) = idx(mask,6);
    
    % Adaptive scaling factors with dynamic balance
    F1 = 0.6 + 0.2 * rand(NP,1);  % Stronger elite guidance
    F2 = 0.4 * (1 - w_f) .* rand(NP,1);
    F3 = 0.4 * w_c .* rand(NP,1);
    rand_pert = 0.05 * randn(NP,D) .* (1 - w_f) .* (1 - w_c);
    
    % Mutation components with vectorized operations
    elite_dir = repmat(elite, NP, 1) - popdecs;
    fit_diff = popdecs(r1,:) - popdecs(r2,:);
    con_diff = popdecs(r3,:) - popdecs;
    
    % Generate offspring with balanced components
    offspring = popdecs + repmat(F1,1,D).*elite_dir + ...
                repmat(F2,1,D).*fit_diff .* repmat(w_f,1,D) + ...
                repmat(F3,1,D).*con_diff .* repmat(w_c,1,D) + ...
                rand_pert;
    
    % Enhanced boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    % Adaptive reflection based on weights
    reflect_factor = 0.5 + 0.5*repmat(w_f.*w_c,1,D);
    offspring = offspring.*(~below & ~above) + ...
                (lb_rep + reflect_factor.*(lb_rep - offspring)).*below + ...
                (ub_rep - reflect_factor.*(offspring - ub_rep)).*above;
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end