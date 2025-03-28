% MATLAB Code
function [offspring] = updateFunc502(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Identify elite solution (best feasible or least infeasible)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Calculate fitness weights using exponential ranking
    [~, rank_order] = sort(popfits);
    ranks = zeros(size(popfits));
    ranks(rank_order) = 1:NP;
    w_fit = exp(-3 * ranks/NP)';  % Less aggressive weighting
    
    % Calculate constraint weights
    abs_cons = abs(cons);
    max_con = max(abs_cons);
    if max_con == 0
        w_con = zeros(size(cons));
    else
        w_con = 1 - exp(-3 * abs_cons/max_con);  % Smoother transition
    end
    
    % Adaptive scaling factors
    F1 = 0.5 + 0.3 * rand(NP,1);  % Strong elite guidance
    F2 = 0.3 + 0.3 * rand(NP,1);  % Fitness-directed exploration
    F3 = 0.1 + 0.2 * rand(NP,1);  % Constraint-aware correction
    F4 = 0.05 * (1 - w_fit) .* (1 - w_con);  % Adaptive perturbation
    
    % Generate random indices (vectorized)
    [~, idx] = sort(rand(NP, NP), 2);
    r1 = idx(:,1);
    r2 = idx(:,2);
    r3 = idx(:,3);
    r4 = idx(:,4);
    
    % Ensure indices are different from current index
    mask = r1 == (1:NP)'; r1(mask) = r2(mask);
    mask = r2 == (1:NP)'; r2(mask) = r3(mask);
    mask = r3 == (1:NP)'; r3(mask) = r4(mask);
    
    % Mutation components
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_fit = popdecs(r1,:) - popdecs(r2,:);
    diff_con = popdecs(r3,:) - popdecs(r4,:);
    rand_pert = randn(NP,D) .* repmat((ub-lb)/10, NP, 1);
    
    % Apply weights
    diff_fit = diff_fit .* repmat(w_fit, 1, D);
    diff_con = diff_con .* repmat(w_con, 1, D);
    
    % Generate offspring with adaptive components
    offspring = popdecs + repmat(F1,1,D).*elite_dir + ...
                repmat(F2,1,D).*diff_fit + ...
                repmat(F3,1,D).*diff_con + ...
                repmat(F4,1,D).*rand_pert;
    
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