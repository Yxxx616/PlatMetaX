% MATLAB Code
function [offspring] = updateFunc506(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-10;
    
    % Elite selection with feasibility consideration
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        elite = popdecs(temp(elite_idx), :);
    else
        % Combine normalized fitness and constraints
        norm_fit = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
        norm_con = abs(cons) / (max(abs(cons)) + eps);
        combined = 0.7*norm_con + 0.3*norm_fit;
        [~, elite_idx] = min(combined);
        elite = popdecs(elite_idx, :);
    end
    
    % Compute adaptive weights
    min_fit = min(popfits);
    max_fit = max(popfits);
    w_f = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    abs_cons = abs(cons);
    max_con = max(abs_cons);
    w_c = abs_cons / (max_con + eps);
    
    % Generate unique random indices
    idx = zeros(NP, 3);
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        idx(i,:) = candidates(randperm(length(candidates), 3));
    end
    r1 = idx(:,1);
    r2 = idx(:,2);
    r3 = idx(:,3);
    
    % Dynamic scaling factors
    F1 = 0.8 + 0.2 * rand(NP,1);  % Strong elite guidance
    F2 = 0.6 * (1 - w_f) .* (0.3 + 0.7*rand(NP,1));
    F3 = 0.6 * w_c .* (0.3 + 0.7*rand(NP,1));
    
    % Small random perturbation
    rand_pert = 0.05 * randn(NP,D) .* (1 - w_f) .* (1 - w_c);
    
    % Vectorized mutation operation
    elite_dir = repmat(elite, NP, 1) - popdecs;
    fit_diff = popdecs(r1,:) - popdecs(r2,:);
    con_diff = popdecs(r3,:) - popdecs;
    
    offspring = popdecs + repmat(F1,1,D).*elite_dir + ...
                repmat(F2,1,D).*fit_diff .* repmat(w_f,1,D) + ...
                repmat(F3,1,D).*con_diff .* repmat(w_c,1,D) + ...
                rand_pert;
    
    % Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    reflect_factor = 0.2 + 0.8*repmat(w_f.*w_c,1,D);
    offspring = offspring.*(~below & ~above) + ...
                (lb_rep + reflect_factor.*(lb_rep - offspring)).*below + ...
                (ub_rep - reflect_factor.*(offspring - ub_rep)).*above;
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end