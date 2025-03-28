% MATLAB Code
function [offspring] = updateFunc463(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Calculate feasibility ratio and identify feasible solutions
    feasible_mask = cons <= 0;
    alpha = sum(feasible_mask) / NP;
    
    % Identify elite solution (best feasible or least infeasible)
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Calculate feasible center (mean of feasible solutions)
    if any(feasible_mask)
        feas_center = mean(popdecs(feasible_mask, :), 1);
    else
        feas_center = mean(popdecs, 1);
    end
    
    % Normalize constraint violations [0,1]
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    
    % Adaptive scaling factors
    F1 = 0.5 * (1 - alpha);
    F2 = 0.3 * alpha;
    F3 = 0.4 * alpha * (1 - norm_cons);
    F4 = 0.6 * (1 - alpha) * norm_cons;
    
    % Expand to D dimensions
    F1 = repmat(F1, 1, D);
    F2 = repmat(F2, 1, D);
    F3 = repmat(F3, 1, D);
    F4 = repmat(F4, 1, D);
    
    % Generate random indices (ensure distinct)
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    r3 = randi(NP, NP, 1);
    
    invalid = (r1 == (1:NP)') | (r2 == (1:NP)') | (r3 == (1:NP)');
    while any(invalid)
        r1(invalid) = randi(NP, sum(invalid), 1);
        r2(invalid) = randi(NP, sum(invalid), 1);
        r3(invalid) = randi(NP, sum(invalid), 1);
        invalid = (r1 == (1:NP)') | (r2 == (1:NP)') | (r3 == (1:NP)');
    end
    
    % Direction vectors
    elite_dir = repmat(elite, NP, 1) - popdecs;
    feas_dir = repmat(feas_center, NP, 1) - popdecs;
    diff_dir = popdecs(r1,:) - popdecs(r2,:);
    rand_dir = popdecs(r3,:) - popdecs;
    
    % Combined mutation with constraint awareness
    offspring = popdecs + F1.*elite_dir + F2.*feas_dir + ...
                F3.*diff_dir + F4.*rand_dir;
    
    % Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection probability based on feasibility
    reflect_prob = 0.3 + 0.4 * (1 - norm_cons);
    reflect_mask = rand(NP,D) < repmat(reflect_prob, 1, D);
    
    % Apply reflection where needed
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
        (2*lb_rep - offspring) .* (mask_low & reflect_mask) + ...
        (2*ub_rep - offspring) .* (mask_high & reflect_mask);
    
    % Random reinitialization for remaining out-of-bounds
    still_violating = (offspring < lb_rep) | (offspring > ub_rep);
    rand_vals = lb_rep + (ub_rep-lb_rep).*rand(NP,D);
    offspring = offspring .* ~still_violating + rand_vals .* still_violating;
end