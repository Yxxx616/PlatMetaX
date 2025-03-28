% MATLAB Code
function [offspring] = updateFunc498(popdecs, popfits, cons)
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
    w_fit = exp(-5 * ranks/NP)';
    
    % Calculate constraint weights
    abs_cons = abs(cons);
    max_con = max(abs_cons);
    if max_con == 0
        w_con = zeros(size(cons));
    else
        w_con = 1 - exp(-5 * abs_cons/max_con);
    end
    
    % Adaptive scaling factors
    F1 = 0.8;  % Strong elite guidance
    F2 = 0.2 + 0.4 * rand(NP,1);  % Fitness-directed exploration
    F3 = 0.1 + 0.2 * rand(NP,1);  % Constraint-aware correction
    F4 = 0.1;  % Adaptive perturbation
    
    % Generate random indices (vectorized)
    [~, idx] = sort(rand(NP, NP), 2);
    mask = idx ~= (1:NP)';
    r1 = zeros(NP,1); r2 = zeros(NP,1); r3 = zeros(NP,1); r4 = zeros(NP,1);
    for i = 1:NP
        candidates = find(mask(i,:));
        r1(i) = candidates(1);
        r2(i) = candidates(2);
        r3(i) = candidates(3);
        r4(i) = candidates(4);
    end
    
    % Mutation components
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_fit = popdecs(r1,:) - popdecs(r2,:);
    diff_con = popdecs(r3,:) - popdecs(r4,:);
    rand_pert = randn(NP,D) .* repmat((ub-lb)/20, NP, 1);
    
    % Apply weights
    diff_fit = diff_fit .* repmat(w_fit, 1, D);
    diff_con = diff_con .* repmat(w_con, 1, D);
    
    % Generate offspring with adaptive components
    offspring = popdecs + F1.*elite_dir + ...
                repmat(F2,1,D).*diff_fit + ...
                repmat(F3,1,D).*diff_con + ...
                F4.*rand_pert;
    
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